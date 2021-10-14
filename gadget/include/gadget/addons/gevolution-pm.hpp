#pragma once
#ifdef GEVOLUTION_PM

#include <gevolution/newtonian_pm.hpp>
#include <gevolution/gr_pm.hpp>
#include <gevolution/Particles_gevolution.hpp>
#include <gevolution/background.hpp>
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
    MyFloat mass;
    std::array<MyFloat,3> Pos;
    std::array<MyFloat,3> Vel;
    std::array<MyFloat,3> Acc;
   
    particle_t(const gevolution::particle & gev_p):
        ID{gev_p.ID},
        mass{gev_p.mass},
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
        ar & mass;
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
        gev_part.mass = mass;
        for(int i=0;i<3;++i)
        {
            gev_part.pos[i]=Pos[i];
            gev_part.vel[i]=Vel[i];
            gev_part.acc[i]=Acc[i];
        }
        return gev_part;   
    }
};

class base_pm
{
    protected:
    ::LATfield2::Lattice lat{};
    
    double Mass_conversion{1}, Pos_conversion{1}, Vel_conversion{1},
           Acc_conversion{1};
    gevolution::cosmology cosmo;
    
    latfield_handler latfield;
    std::unique_ptr< particle_handler> Sp;
    std::stringstream my_log;
    
    // this pointer contains meaningful data only if the currect process is
    // "latfield.active()", ie. if it holds Particle Mesh data. Notice that all
    // processes participate in the construction of the Particle Mesh, that's
    // unavoidable because every process has its own fraction of the particles
    // of the simulation.
    std::unique_ptr< gevolution::Particles_gevolution> pcls_cdm;
    int _size;
    MyFloat _boxsize;
    int _sample_p_correction;
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
        base_pm& pm;
        std::unordered_map<MyIDType,int> origin; 
        
        public:
        gadget_domain_t(base_pm& ref):
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
            // C++ magic: we use the for_each() method from LATfield to iterate
            // over particles.
            pm.pcls_cdm->for_each
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
        base_pm& pm;
            
        // tag particles that come from non-active processes
        std::unordered_map<MyIDType,bool> from_twin; 
        
        public:
        latfield_domain_t(base_pm& ref):
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
    const ::LATfield2::Lattice& lattice() const
    {
        return lat;
    }
    
    std::string get_log()
    {
        std::string log_mes = my_log.str();
        my_log.str("");
        return log_mes;
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
        gadget::global_data_all_processes gadget_data, 
        double asmth)
    {
        // boxsize in units of Mpc/h
        _boxsize = gadget_data.BoxSize * gadget_data.UnitLength_in_cm 
            / (1e6 * PARSEC);
            
        // asmth in units of Mpc/h
        asmth *= gadget_data.UnitLength_in_cm
            / (1e6 * PARSEC);
        Asmth2 = asmth*asmth;
        
        cosmo.h = gadget_data.HubbleParam;
        cosmo.fourpiG = 1.5 * _boxsize * _boxsize / cosmo.C_SPEED_OF_LIGHT / cosmo.C_SPEED_OF_LIGHT;
        
        cosmo.Omega_fld = cosmo.wa_fld = cosmo.w0_fld = 0;
        cosmo.num_ncdm = 0;
        
        // TODO: check if these are correct
        cosmo.Omega_rad = 0;
        cosmo.Omega_b = gadget_data.OmegaBaryon;
        cosmo.Omega_m = gadget_data.Omega0;
        cosmo.Omega_Lambda = gadget_data.OmegaLambda;
        cosmo.Omega_cdm = cosmo.Omega_m - cosmo.Omega_b; 
        
        // Pos_conversion  = 1.0/gadget_data.BoxSize;
        Pos_conversion = 1.0;
        Vel_conversion  = 1.0/gadget_data.c;
        Mass_conversion = 8*M_PI*gadget_data.G/3
            /gadget_data.BoxSize/gadget_data.BoxSize/gadget_data.BoxSize/gadget_data.Hubble/gadget_data.Hubble;
        Acc_conversion = gadget_data.c * gadget_data.c 
           / gadget_data.BoxSize / gadget_data.G;
        
        // old way
        // cosmo.fourpiG = 1;
        // Mass_conversion = 1;
        // Acc_conversion = 4*pi / gadget_data.BoxSize / gadget_data.BoxSize;
        
        
        _sample_p_correction = gadget_data.SamplingCorrection;
        Sp.reset(Sp_ptr);
        
        if(latfield.active())
            // executed by active processes only
        {
        // for the lack of a good constructor and reset function
        // we do this
            gevolution::particle_info pinfo;
            std::strcpy(pinfo.type_name,"gevolution::particle");
            pinfo.mass = 1.0;// this is ignored, particles have individual mass
            pinfo.relativistic=false;
            std::array<double,3> box{1.,1.,1.};
            pcls_cdm->initialize(
                pinfo,
                gevolution::particle_dataType{},
                &lattice(),
                box.data());
        }
    }
    
    
    int sampling_correction_order()const
    /*
        order of the correction to sampling in k-space.
        eg. p = 2 for CIC
    */
    {
        return _sample_p_correction;
    }
    
    int signed_mode(int k)const
    {
        return k >= size() / 2 ? k - size() : k;
    }
   
    virtual ~base_pm()
    {}
    base_pm(MPI_Comm raw_com,int Ngrid):
        latfield(raw_com),
        _size(Ngrid)
    {
        if(latfield.active())
        {
            // TODO: do not repeat this initialization here, use a single lattice
            lat.initialize(
                /* dims        = */ 3,
                /* size        = */ Ngrid,
                /* ghost cells = */ 2);
        }
    }
};

class newtonian_pm : 
    public base_pm
{
    using gev_pm = ::gevolution::newtonian_pm< ::gevolution::Cplx,::gevolution::Particles_gevolution >;
    using base_pm::latfield;
    
    // WARNING: gev_pm is only accessible to those processes participating in
    // LATfield, ie. latfield.active()
    
    std::unique_ptr<gev_pm> gev_pm_ptr;
    
    public:
    using base_pm::size;
    
    newtonian_pm(MPI_Comm raw_com,int Ngrid):
        base_pm(raw_com,Ngrid)
    {
        if(latfield.active())
        {
            gev_pm_ptr.reset(new gev_pm{Ngrid});
            pcls_cdm.reset(new gevolution::Particles_gevolution{} );
        }
    }
    
    void pmforce_periodic(int,int*, double a)
    {
        my_log << "calling " << __PRETTY_FUNCTION__ << "\n"; 
        
        // tag particles index in the handler
        std::unordered_map<MyIDType,int> Sp_index; 
        
        // load particles into buffer
        P_buffer.resize(Sp->size());
        for(auto i=0U;i<P_buffer.size();++i)
        {
            auto & p = P_buffer[i];
            p.ID = Sp->get_id(i);
            p.mass = Sp->get_mass(i) * Mass_conversion;
            p.Vel= Sp->get_velocity(i);
            p.Pos= Sp->get_position(i);
            
            for(int k=0;k<3;++k)
            {
                p.Vel[k] *= Vel_conversion;
                p.Pos[k] *= Pos_conversion;
            }
            
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
                
                gev_pm_ptr -> clear_sources();
                gev_pm_ptr -> sample(*pcls_cdm,a);
                auto [mean_m,mean_p,mean_v] = gev_pm_ptr -> test_velocities(*pcls_cdm);
                my_log << "mean     mass: " << mean_m << "\n";
                my_log << "mean sqr(pos): " << mean_p << "\n";
                my_log << "mean sqr(vel): " << mean_v << "\n";
        
                gev_pm_ptr -> compute_potential(cosmo.fourpiG, a);
                
                // sampling spline correction order p (p=2 CIC)
                ::gevolution::apply_filter_kspace(
                     *gev_pm_ptr,
                     [this](std::array<int,3> mode)
                     {
                         double factor{1.0};
                         for(int i=0;i<3;++i)
                         if(mode[i]){
                             double phase = signed_mode(mode[i]) * pi / size();
                             factor *= phase / std::sin(phase);
                         }
                         return std::pow(factor,sampling_correction_order());
                     });
                
                // smoothing the field at the Asmth scale
                ::gevolution::apply_filter_kspace(
                    *gev_pm_ptr,
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
                
                gev_pm_ptr -> compute_forces(*pcls_cdm,1.0,a);
            }
        }
        
        // set the accelerations
        const MyFloat conversion_factor = a * Acc_conversion;
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
    
    ~newtonian_pm() override
    {}
};

// class relativistic_pm : public base_pm
// {
//     using base_pm::latfield;
//     
//     public:
//     relativistic_pm(MPI_Comm raw_com,int Ngrid):
//         base_pm(raw_com,Ngrid)
//     {
//         if(latfield.active())
//         {
//             pcls_cdm.reset(new gevolution::Particles_gevolution{} );
//         }
//     }
//     
//     
//     /* TODO: remove the lines of code that repeat */
//     void pmforce_periodic(int,int*, double a /* scale factor */) override
//     {
//         my_log << "calling " << __PRETTY_FUNCTION__ << "\n"; 
//         // tag particles index in the handler
//         std::unordered_map<MyIDType,int> Sp_index; 
//         
//         // load particles into buffer
//         P_buffer.resize(Sp->size());
//         for(auto i=0U;i<P_buffer.size();++i)
//         {
//             auto & p = P_buffer[i];
//             p.ID = Sp->get_id(i);
//             p.mass = Sp->get_mass(i) * Mass_conversion;
//             p.Vel= Sp->get_velocity(i); // TODO: convert velocity to momentum
//             p.Pos= Sp->get_position(i); // in units of the boxsize
//             
//             for(int k=0;k<3;++k)
//             {
//                 p.Vel[k] *= Vel_conversion;
//                 p.Pos[k] *= Pos_conversion;
//             }
//             
//             Sp_index[p.ID] = i;
//         }
//         #ifndef NDEBUG
//         std::size_t start_hash = hash_ids();
//         #endif
//         
//         {
//             // send particles to active processes
//             // on destruction particles will be sent back
//             latfield_domain_t D_lat{*this}; 
//             
//             if(latfield.active())
//             {
//                 // send particles from gadget's domain to latfield's 
//                 // on destruction particles will be sent back
//                 gadget_domain_t D_gad{*this}; 
//                 
//                 // update pcls_cdm from P_buffer
//                 pcls_cdm->clear(); // remove existing particles, we start fresh
//                 bool success = true;
//                 for(const auto &p : P_buffer)
//                     success &= pcls_cdm->addParticle_global(gevolution::particle(p));
//                 assert(success);
//                 
//                 gev_pm->clear_sources(); // OK
//                 gev_pm->sample(*pcls_cdm,a); 
//                 auto [mean_m,mean_p,mean_v] = gev_pm->test_velocities(*pcls_cdm);
//                 my_log << "mean     mass: " << mean_m << "\n";
//                 my_log << "mean sqr(pos): " << mean_p << "\n";
//                 my_log << "mean sqr(vel): " << mean_v << "\n";
//                 
//                 // TODO: compute dtau
//                 gev_pm->compute_potential(
//                     a,
//                     gevolution::Hconf(a,cosmo),
//                     cosmo.fourpiG,
//                     /* dtau = */ 1.0,
//                     cosmo.Omega_cdm + cosmo.Omega_b + bg_ncdm (a, cosmo)
//                     ); 
//                
//                 gev_pm->compute_forces(*pcls_cdm,a);
//                 
//             }
//         }
//         
//         // set the accelerations
//         // TODO: correct conversion factor
//         const MyFloat conversion_factor 
//             = 4 * pi / boxsize() / boxsize() / Mass_conversion;
//         for(auto& p : P_buffer)
//         {
//             const int i= Sp_index.at(p.ID);
//             
//             // TODO: convert d(a q)/dt to d(a u)/dt
//             for(auto & ax : p.Acc)
//                 ax *= conversion_factor;
//             
//             Sp->set_acceleration(i,p.Acc);
//         }
//         #ifndef NDEBUG
//         std::size_t end_hash = hash_ids();
//         assert(start_hash == end_hash);
//         #endif
//     }
//     
//     ~relativistic_pm() override
//     {}
// };

}

#endif
