#pragma once
#ifdef GEVOLUTION_PM

#include <gevolution/newtonian_pm.hpp>
#include <gevolution/Particles_gevolution.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/container_hash/hash.hpp>

#include <memory>
#include <array>
#include <vector>
#include <sstream>
#include <gadget/addons/latfield_handler.hpp>
#include "gadget/particle_handler.h"  // particle_handler

namespace gadget::gevolution_api
{

struct particle_t 
{
    MyIDType ID;
    std::array<MyFloat,3> Pos;
    std::array<MyFloat,3> Vel;
    std::array<MyFloat,3> Acc;
   
    particle_t(const gevolution::particle & gev_p):
        ID{gev_p.ID},
        Pos{gev_p.pos[0],gev_p.pos[1],gev_p.pos[2]},
        Vel{gev_p.vel[0],gev_p.vel[1],gev_p.vel[2]},
        Acc{gev_p.acc}
    {}
   
    particle_t(MyIDType id):
        ID{id}
    {}
    
    particle_t() = default;
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /*version*/)
    {
        ar & ID;
        ar & Pos;
        ar & Vel;
        ar & Acc;
    }
    
    bool operator < (const particle_t & that)const
    {
        return ID < that.ID;
    }
    
    operator gevolution::particle() const
    {
        // no c++ structures, hence we need to construct by hand
        gevolution::particle gev_part; 
        gev_part.ID = ID;
        for(int i=0;i<3;++i)
        {
            gev_part.pos[i]=Pos[i];
            gev_part.vel[i]=Vel[i];
            gev_part.acc[i]=Acc[i];
        }
        return gev_part;   
    }
};

class newtonian_pm
{
    latfield_handler latfield;
    std::unique_ptr< particle_handler> Sp;
    
    // this pointer contains meaningful data only if the currect process is
    // "latfield.active()", ie. if it holds Particle Mesh data. Notice that all
    // processes participate in the construction of the Particle Mesh, that's
    // unavoidable because every process has its own fraction of the particles
    // of the simulation.
    std::unique_ptr< gevolution::newtonian_pm > gev_pm;
    std::unique_ptr< gevolution::Particles_gevolution> pcls_cdm;
    int _size;
    MyFloat _boxsize, Mass;
    double Asmth2; // smoothing scale (in gadget length units) squared
    static constexpr MyFloat pi = boost::math::constants::pi<MyFloat>();
    std::vector<particle_t> P_buffer;
    
    
    class gadget_domain_t
    /*
        helper class
        it handles in a RAII way the three step process of PM computations by
        the partecipating mpi ranks:
        1. ctor: exchange particle data to adjust for the domain decomposition
        in gadget and gevolution.
        2. whatever the active mpi ranks do with the particles in their domain.
        3. dtor: exchange particle data back
        
        only valid for active processes!
    */
    {
        newtonian_pm& pm;
        std::unordered_map<MyIDType,int> origin; 
        
        public:
        gadget_domain_t(newtonian_pm& ref):
            pm{ref}
        {
            // send to correct process based on position
            std::vector< std::vector<particle_t> > P_sendrecv(pm.latfield.com_pm.size());
            
            for(const auto & p : pm.P_buffer)
            {
                // LATfield two-step way to determine a particle's process
                int proc_rank[2];
                pm.pcls_cdm->getPartProcess( gevolution::particle(p), proc_rank) ;
                const int destination = pm.latfield.grid2world(proc_rank[0],proc_rank[1]);
                P_sendrecv[destination].emplace_back(p);
            }
            pm.P_buffer.clear();
            
            boost::mpi::all_to_all(pm.latfield.com_pm,P_sendrecv,P_sendrecv);
            
            for(auto i=0U;i<P_sendrecv.size();++i)
            {
                auto & v = P_sendrecv[i];
                for(const auto & p : v)
                {
                    origin[p.ID] = i;
                    pm.P_buffer.emplace_back(p);
                }
                v.clear();
            }
        }
        ~gadget_domain_t()
        {
            std::vector< std::vector<particle_t> > P_sendrecv(pm.latfield.com_pm.size());
            pm.P_buffer.clear();
            // C++ magic: we use the updateVel() method from LATfield to iterate
            // over particles. Clearly this method has a misleading name.
            pm.pcls_cdm->updateVel
            (
                [this,&P_sendrecv]
                (const gevolution::particle & gev_part, LATfield2::Site)
                {
                   particle_t p{gev_part};
                   const int destination = origin[p.ID];
                   P_sendrecv[destination].emplace_back( std::move(p));
                   return 0.0;
                }
            );
            // send back to correct process
            boost::mpi::all_to_all(pm.latfield.com_pm,P_sendrecv,P_sendrecv);
            
            // update P_buffer from pcls_cdm
            for(auto i=0U;i<P_sendrecv.size();++i)
            {
                const auto & v = P_sendrecv[i];
                for(const auto & p : v)
                    pm.P_buffer.emplace_back(p);
            }
        }
    };
    
    class latfield_domain_t
    /*
        helper class
        it handles in a RAII way the three step process of PM computation:
        1. ctor: exchange particle data between process that partecipate and
        don't
        2. whatever the active mpi ranks do.
        3. dtor: exchange particle data back
    */
    {
        newtonian_pm& pm;
            
        // tag particles that come from non-active processes
        std::unordered_map<MyIDType,bool> from_twin; 
        
        public:
        latfield_domain_t(newtonian_pm& ref):
            pm{ref}
        // exchange particles forward to active processes
        {
            for(const auto& p: pm.P_buffer)
            {
                from_twin[p.ID] = false;
            }
            
            if(pm.latfield.active())
            {
                if(pm.latfield.has_twin())
                {
                    std::vector<particle_t> P_recv;
                    pm.latfield.com_world.recv(
                        /* source = */ pm.latfield.twin_rank(),
                        /* tag = */ 0,
                        /* value = */ P_recv );    
                    std::copy(P_recv.begin(),P_recv.end(),std::back_inserter(pm.P_buffer));
                    
                    for(const auto & p : P_recv)
                        from_twin[p.ID] = true;
                }else
                {
                   // std::cout << "rank = " << latfield.this_rank() << " has no twin"
                   // << std::endl;
                }
            }
            else
            {
                pm.latfield.com_world.send(
                    /* dest = */ pm.latfield.twin_rank(),
                    /* tag = */ 0,
                    /* value = */ pm.P_buffer );    
                pm.P_buffer.clear();
            }
        }
        
        ~latfield_domain_t()
        // exchange particles backwards from active processes
        {
            std::sort(pm.P_buffer.begin(),pm.P_buffer.end(),
                [this](const particle_t& a, const particle_t& b)
                {
                    return from_twin[a.ID] < from_twin[b.ID];
                });
            std::vector<particle_t> P_send;
            while(not pm.P_buffer.empty())
            {
                if(from_twin[pm.P_buffer.back().ID])
                {
                    P_send.push_back( pm.P_buffer.back() );
                    pm.P_buffer.pop_back();
                }else
                    break;
            }
            if(pm.latfield.active())
            {
                if(pm.latfield.has_twin())
                {
                    pm.latfield.com_world.send(
                        /* source = */ pm.latfield.twin_rank(),
                        /* tag = */ 1,
                        /* value = */ P_send );    
                }else
                {
                   // has no twin
                }
            }
            else
            {
                std::vector<particle_t> P_recv;
                pm.latfield.com_world.recv(
                    /* dest = */ pm.latfield.twin_rank(),
                    /* tag = */ 1,
                    /* value = */ P_recv );    
                std::copy(P_recv.begin(),P_recv.end(),std::back_inserter(pm.P_buffer));
            }
            
        }
    };
    
    std::size_t hash_ids()
    {
        std::size_t seed{1};
        std::sort(P_buffer.begin(),P_buffer.end(),
            [](const particle_t& a, const particle_t & b)
            {
                return a.ID < b.ID;
            });
        for(const auto& p: P_buffer)
        {
            boost::hash_combine(seed,p.ID);
        }
        return seed;
    }
    
    public:
    void set_mass(MyFloat M)
    {
        Mass = M;
    }
    
    newtonian_pm(MPI_Comm raw_com,int Ngrid):
        latfield(raw_com),
        _size(Ngrid)
    {
        if(latfield.active())
        {
            gev_pm.reset(new gevolution::newtonian_pm{Ngrid} );    
            pcls_cdm.reset(new gevolution::Particles_gevolution{} );
        }
    }
    
    int size()const{return _size;}
    MyFloat k_fundamental() const{return 2*pi/boxsize();}
    MyFloat boxsize() const {return _boxsize;}
    
    void calculate_power_spectra(int num, char *OutputDir)
    {
        // ? 
    }
    
    void pm_init_periodic(
        particle_handler *Sp_ptr, 
        double in_boxsize /* */,
        double M /* particle mass */,
        double asmth)
    {
        _boxsize = in_boxsize;
        Mass = M;
        Sp.reset(Sp_ptr);
        Asmth2 = asmth*asmth;
        
        if(latfield.active())
            // executed by active processes only
        {
        // for the lack of a good constructor and reset function
        // we do this
            gevolution::particle_info pinfo;
            std::strcpy(pinfo.type_name,"gevolution::particle");
            pinfo.mass = 1.0;// unit of mass such that particles have mass = 1!!
            pinfo.relativistic=false;
            std::array<double,3> box{1.,1.,1.};
            pcls_cdm->initialize(
                pinfo,
                gevolution::particle_dataType{},
                &(gev_pm->lattice()),
                box.data());
        }
    }
    
    int signed_mode(int k)const
    {
        return k >= size() / 2 ? k - size() : k;
    }
    
    void pmforce_periodic(int,int*)
    {
        // tag particles index in the handler
        std::unordered_map<MyIDType,int> Sp_index; 
        
        // load particles into buffer
        P_buffer.resize(Sp->size());
        for(auto i=0U;i<P_buffer.size();++i)
        {
            auto & p = P_buffer[i];
            p.ID = Sp->get_id(i);
            p.Vel= Sp->get_velocity(i); // TODO: unit conversion
            p.Pos= Sp->get_position(i); // in units of the boxsize
            
            Sp_index[p.ID] = i;
        }
        #ifndef NDEBUG
        std::size_t start_hash = hash_ids();
        #endif
        
        {
            // send particles to active processes
            // on destruction particles will be sent back
            latfield_domain_t D_lat{*this}; 
            
            if(latfield.active())
            {
                // send particles from gadget's domain to latfield's 
                // on destruction particles will be sent back
                gadget_domain_t D_gad{*this}; 
                
                // update pcls_cdm from P_buffer
                pcls_cdm->clear(); // remove existing particles, we start fresh
                bool success = true;
                for(const auto &p : P_buffer)
                    success &= pcls_cdm->addParticle_global(gevolution::particle(p));
                assert(success);
                
                gev_pm->sample(*pcls_cdm);
                
                gev_pm->update_kspace();
                // k-space begin
                
                gev_pm->solve_poisson_eq();
                
                // twice CIC correction,
                // gev_pm->apply_filter_kspace( 
                //     [this](std::array<int,3> mode)
                //     {
                //         double factor{1.0};
                //         for(int i=0;i<3;++i)
                //         if(mode[i]){
                //             double phase = signed_mode(mode[i]) * pi / size();
                //             factor *= phase / std::sin(phase);
                //         }
                //         return factor*factor*factor*factor;
                //     });
                
                // smoothing the field at the Asmth scale
                gev_pm->apply_filter_kspace( 
                    [this](std::array<int,3> mode)
                    {
                        double k2{0.0};
                        for(int i=0;i<3;++i)
                        {
                            double ki = signed_mode(mode[i]) * k_fundamental();
                            k2 += ki*ki;
                        }
                        return std::exp( - Asmth2 * k2);
                    });
                
                // k-space end
                gev_pm->update_rspace();
                
                gev_pm->compute_forces(*pcls_cdm,/* fourpiG = */ 1.0 ); 
                
            }
        }
        
        // set the accelerations
        const MyFloat conversion_factor 
            = 4 * pi * Mass / boxsize() / boxsize();
        for(auto& p : P_buffer)
        {
            const int i= Sp_index.at(p.ID);
            
            for(auto & ax : p.Acc)
                ax *= conversion_factor;
            
            Sp->set_acceleration(i,p.Acc);
        }
        #ifndef NDEBUG
        std::size_t end_hash = hash_ids();
        assert(start_hash == end_hash);
        #endif
    }
    
    void compute_forces()
        // only executed by active processes
    {
    }
};

}

#endif
