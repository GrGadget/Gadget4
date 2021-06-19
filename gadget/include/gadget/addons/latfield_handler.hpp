#pragma once
#ifdef GEVOLUTION_PM

#include <LATfield2.hpp>
#include <boost/mpi/communicator.hpp>
#include <mpi.h>
#include <iostream>

/*
    This is a class to handle LATfield in a RAI way.
*/

namespace gadget::gevolution_api
{

class latfield_handler
{
    int proc_color,twin_proc;
    
    // latfield proc cartesian communicator
    int nx,ny,ntot;
    
    static int highest_bit(int n)
    {
        int p=-1;
        while(n)
        {
            ++p;
            n >>= 1;
        }
        return p;
    }
    
    public:
    boost::mpi::communicator com_world;
    boost::mpi::communicator com_pm;
    
    latfield_handler(MPI_Comm raw_com):
       com_world(raw_com,boost::mpi::comm_duplicate)
    {
        const int b = highest_bit(com_world.size());
        proc_color = com_world.rank() < (1<<b) ? 0 : 1;
        com_pm = com_world.split( proc_color );   
        
        const int bx = b/2, by = b-bx;
        nx = 1 << bx , ny = 1 << by;
        ntot = nx * ny;
        
        int p1 = com_world.rank() % ntot,
            p2 = p1 + ntot; 
        if(p2 < com_world.size())
            twin_proc = active() ? p2 : p1;
        else
            twin_proc = com_world.rank();// this proc is alone
        
        if(active())
        {
            LATfield2::parallel.initialize(MPI_Comm(com_pm),nx,ny);
            if(com_pm.rank()==0)
                std::cout << 
                    "LATfield2 initialized with n = " 
                    << com_pm.size() 
                    << " subprocesses."
                    << " nx = " << nx << ", ny = " << ny << '\n';
        }
        
    }
    
    int grid2world(int n,int m)const
    {
        return LATfield2::parallel.grid2world(n,m);
    }
    
    // this process participates in the PM
    bool active() const { return proc_color == 0;}
    
    int twin_rank() const { return twin_proc; }
    int this_rank() const { return com_world.rank(); }
    
    bool has_twin() const { return twin_proc != com_world.rank();}
};

}
#endif
