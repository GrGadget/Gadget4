#pragma once
#ifdef GEVOLUTION_PM

#include <gevolution/newtonian_pm.hpp>
#include <gevolution/Particles_gevolution.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>

#include <memory>
#include <array>
#include <vector>
#include <gadget/addons/latfield_handler.hpp>
#include "gadget/particle_handler.h"  // particle_handler

// how to initialize particles?
// how to update particles at every time step?

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
    std::unique_ptr< gevolution::newtonian_pm > pm;
    std::unique_ptr< gevolution::Particles_gevolution> pcls_cdm;
    int _size;
    MyFloat boxsize, G_cavendish, Mass;
    static constexpr MyFloat PI = std::acos(-1.0);
    std::vector<particle_t> P_buffer;
    
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
            pm.reset(new gevolution::newtonian_pm{Ngrid} );    
            pcls_cdm.reset(new gevolution::Particles_gevolution{} );
        }
    }
    
    int size()const{return _size;}
    
    void calculate_power_spectra(int num, char *OutputDir)
    {
        // ? 
    }
    
    void pm_init_periodic(
        particle_handler *Sp_ptr, 
        double in_boxsize /* */,
        double G /* Gravitational constant  */,
        double M /* particle mass */, 
        double asmth)
    {
        boxsize = in_boxsize;
        G_cavendish = G;
        Mass = M;
        Sp.reset(Sp_ptr);
        
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
                &(pm->lattice()),
                box.data());
        }
    }
    
    void pmforce_periodic(int, int*)
    {
        // exchange particles forward to active processes
        P_buffer.resize(Sp->size());
        
        // tag particles that come from non-active processes
        std::unordered_map<MyIDType,bool> from_twin; 
        // tag particles index in the handler
        std::unordered_map<MyIDType,int> Sp_index; 
        
        for(auto i=0U;i<P_buffer.size();++i)
        {
            auto & p = P_buffer[i];
            p.ID = Sp->get_id(i);
            p.Vel= Sp->get_velocity(i); // TODO: unit conversion
            
            p.Pos= Sp->get_position(i);
            for(auto & x : p.Pos) x /= boxsize;
            
            Sp_index[p.ID] = i;
            from_twin[p.ID] = false;
        }
        
        //std::cout 
        //    << "rank = " << latfield.this_rank() 
        //    << " buff size = " <<  P_buffer.size() 
        //    << "\n\n" << std::endl;
        
        if(latfield.active())
        {
            if(latfield.has_twin())
            {
                std::vector<particle_t> P_recv;
                latfield.com_world.recv(
                    /* source = */ latfield.twin_rank(),
                    /* tag = */ 0,
                    /* value = */ P_recv );    
                std::copy(P_recv.begin(),P_recv.end(),std::back_inserter(P_buffer));
                // std::cout 
                //     << "rank = " << latfield.this_rank() 
                //     << ", twin rank = "
                //     << latfield.twin_rank() << std::endl;
                
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
            latfield.com_world.send(
                /* dest = */ latfield.twin_rank(),
                /* tag = */ 0,
                /* value = */ P_buffer );    
            P_buffer.clear();
        }
        
        if(latfield.active())
            compute_forces();
        
        // exchange particles backwards from active processes
        {
            std::sort(P_buffer.begin(),P_buffer.end(),
                [&from_twin](const particle_t& a, const particle_t& b)
                {
                    return from_twin[a.ID] < from_twin[b.ID];
                });
            std::vector<particle_t> P_send;
            while(not P_buffer.empty())
            {
                if(from_twin[P_buffer.back().ID])
                {
                    P_send.push_back( P_buffer.back() );
                    P_buffer.pop_back();
                }else
                    break;
            }
            if(latfield.active())
            {
                if(latfield.has_twin())
                {
                    latfield.com_world.send(
                        /* source = */ latfield.twin_rank(),
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
                latfield.com_world.recv(
                    /* dest = */ latfield.twin_rank(),
                    /* tag = */ 1,
                    /* value = */ P_recv );    
                std::copy(P_recv.begin(),P_recv.end(),std::back_inserter(P_buffer));
            }
        }
        // std::cout 
        //     << "rank = " << latfield.this_rank() 
        //     << " final buff size = " <<  P_buffer.size() 
        //     << "\n\n" << std::endl;
        
        const MyFloat conversion_factor 
            = 4 * PI * G_cavendish * Mass / boxsize / boxsize;
        std::cout 
            << "rank = " << latfield.this_rank() 
            << " conversion factor = " <<  conversion_factor
            << " PI = " << PI
            << " G = " << G_cavendish
            << " M = " << Mass
            << " L = " << boxsize
            << "\n\n" << std::endl;
        for(auto& p : P_buffer)
        {
            const int i= Sp_index.at(p.ID);
            
            for(auto & ax : p.Acc)
                ax *= conversion_factor;
            
            Sp->set_acceleration(i,p.Acc);
        }
    }
    
    void compute_forces()
        // only executed by active processes
    {
        // send to correct process based on position
        std::vector< std::vector<particle_t> > P_sendrecv(latfield.com_pm.size());
        
        for(const auto & p : P_buffer)
        {
            // LATfield two-step way to determine a particle's process
            int proc_rank[2];
            pcls_cdm->getPartProcess( gevolution::particle(p), proc_rank) ;
            const int destination = latfield.grid2world(proc_rank[0],proc_rank[1]);
            P_sendrecv[destination].emplace_back(p);
        }
        const size_t init_size = P_buffer.size(); // sanity check
        P_buffer.clear();
        
        boost::mpi::all_to_all(latfield.com_pm,P_sendrecv,P_sendrecv);
        std::unordered_map<MyIDType,int> origin; 
        
        for(auto i=0U;i<P_sendrecv.size();++i)
        {
            auto & v = P_sendrecv[i];
            for(const auto & p : v)
            {
                origin[p.ID] = i;
                P_buffer.emplace_back(p);
            }
            v.clear();
        }
        
        // update pcls_cdm from P_buffer
        pcls_cdm->clear(); // remove existing particles, we start fresh
        bool success = true;
        for(const auto &p : P_buffer)
            success &= pcls_cdm->addParticle_global(gevolution::particle(p));
        assert(success);
        
        
        // construct the Energy-Momentum Tensor
        pm->sample(*pcls_cdm);
        
        // TODO: take into account the smoothing
        pm->compute_potential(); 
        
        pm->compute_forces(*pcls_cdm,/* fourpiG = */ 1.0 ); 
        
        // update P_buffer from pcls_cdm
        P_buffer.clear();
        // C++ magic: we use the updateVel() method from LATfield to iterate
        // over particles. Clearly this method has a misleading name.
        pcls_cdm->updateVel
        (
            [this]
            (const gevolution::particle & gev_part, LATfield2::Site)
            {
               P_buffer.emplace_back(gev_part);
               return 0.0;
            }
        );
        
        // send back to correct process
        for(const auto & p : P_buffer)
        {
            const int destination = origin[p.ID]; // original process
            P_sendrecv[destination].emplace_back(p);
        }
        P_buffer.clear();
        boost::mpi::all_to_all(latfield.com_pm,P_sendrecv,P_sendrecv);
        
        for(auto i=0U;i<P_sendrecv.size();++i)
        {
            const auto & v = P_sendrecv[i];
            for(const auto & p : v)
                P_buffer.emplace_back(p);
        }
        
        if(init_size != P_buffer.size())
        // sanity check
            assert( init_size == P_buffer.size() );
    }
};

}

#endif
