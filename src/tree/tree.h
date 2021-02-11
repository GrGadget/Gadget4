/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  tree.h
 *
 *  \brief declaration of the base class for building oct-trees
 */

#ifndef TREE_H
#define TREE_H

#include "gadgetconfig.h"

#ifndef TREE_NUM_BEFORE_NODESPLIT
#define TREE_NUM_BEFORE_NODESPLIT 3  // daughter nodes are only created if there are more than this number of particles in a node
#endif

#define TREE_MODE_BRANCH 0
#define TREE_MODE_TOPLEVEL 1

#define MAX_TREE_ALLOC_FACTOR 30.0

#define TREE_MAX_ITER 100

#include "../data/mymalloc.h"                 // Mem.
#include "../domain/domain.h"                 // template class domain;
#include "../mpi_utils/shared_mem_handler.h"  // shmem Shmem; global variable

#if MAX_NUMBER_OF_RANKS_WITH_SHARED_MEMORY <= 32
typedef std::uint32_t node_bit_field;
#elif MAX_NUMBER_OF_RANKS_WITH_SHARED_MEMORY <= 64
typedef std::uint64_t node_bit_field;
#else
#error "unsupported MAX_NUMBER_OF_RANKS_WITH_SHARED_MEMORY setting"
#endif

/** The tree node data structure. Nodes points to the actual memory
    allocated for the internal nodes, but is shifted such that
    Nodes[Sp.MaxPart] gives the first allocated node. Note that node
    numbers less than Sp.MaxPart are the leaf nodes that contain a
    single particle, and node numbers >= MaxPart+MaxNodes are "pseudo
    particles" that hang off the toplevel leaf nodes belonging to
    other tasks. These are not represented by this structure. Instead,
    the tree traversal for these are saved in the Nextnode, Prevnode
    and Father arrays, indexed with the node number in the case of
    real particles and by nodenumber-MaxNodes for pseudo
    particles.  */

struct basenode
{
  std::atomic<node_bit_field> flag_already_fetched;

  vector<MyIntPosType> center; /**< geometrical center of node */

  int sibling;
  /** The next node in case the current node needs to be
      opened. Applying nextnode repeatedly results in a pure
      depth-first traversal of the tree. */
  int nextnode;
  /** The parent node of the node. (Is -1 for the root node.) */
  int father;

  int OriginTask; /* MPI rank (in full compute communicator) on which this node and its daughter nodes are natively stored */
  int OriginNode; /* Index of the node on the MPI rank that stores it and its daughter nodes natively */

  unsigned char level; /**< hold the tree level, used to store the side length of node in space efficient way */
  unsigned char sibling_shmrank;
  unsigned char nextnode_shmrank;

  std::atomic_flag access;

  std::atomic<unsigned char> cannot_be_opened_locally;

  // unsigned char cannot_be_opened_locally : 1;
  unsigned char not_empty : 1;
};

struct node_info
{
  int Node;
};

struct data_nodelist
{
  int Task;           /** target process */
  int Index;          /** local index that wants to open this node */
  node_info NodeInfo; /** info about node to be opened on foreign process, as well as perdiodic box offset (needed for Ewald summation
                      algorithm for periodic gravity */
};

template <typename node, typename partset, typename point_data, typename foreign_point_data>
class tree
{
  /* class for oct tree */

  struct index_data
  {
    int p;
    int subnode;
    // FIXME: define a comparison operation
  };
  // FIXME: delete this function
  static inline bool compare_index_data_subnode(const index_data &a, const index_data &b) { return a.subnode < b.subnode; }
  
    
 protected:
  
  int *Father;
  int *Nextnode;
  int *NodeIndex;
  
  partset *Tp;
  domain<partset> *D;
  
  node *TopNodes;
  node *Nodes;
  node *Foreign_Nodes;
  
  foreign_point_data *Foreign_Points;
  
  ptrdiff_t *TreeP_offsets;
  ptrdiff_t *TreePS_offsets;
  ptrdiff_t *TreeSphP_offsets;
  ptrdiff_t *TreeForeign_Points_offsets;
  ptrdiff_t *TreeForeign_Nodes_offsets;
  
  void **TreeSharedMemBaseAddr;
  
  int *IndexList;
  int *ResultIndexList;
  
  int *Send_offset;
  int *Send_count;
  int *Recv_count;
  int *Recv_offset;
  
  int MaxPart;
  int MaxNodes;
  int NumNodes;
  int NumPartExported;
  
  int NumForeignNodes;  // number of imported foreign tree nodes
  int MaxForeignNodes;
  int NumForeignPoints;  // number of imported foreign particles to allow completion of local tree walks
  int MaxForeignPoints;
  
  struct fetch_data
  {
    int NodeToOpen;
    int ShmRank;
    int GhostRank;
  };
  fetch_data *StackToFetch;
  
  struct workstack_data
  {
    int Target;
    int Node;
    int ShmRank;
    int MinTopLeafNode;
    
    // FIXME: implement a comparison method
  };

  workstack_data *WorkStack;
  // FIXME: remove this function
  static bool compare_workstack(const workstack_data &a, const workstack_data &b)
  {
    if(a.MinTopLeafNode < b.MinTopLeafNode)
      return true;
    if(a.MinTopLeafNode > b.MinTopLeafNode)
      return false;

    return a.Target < b.Target;
  }
  
 public:
  typedef decltype(Tp->P) pdata;
  
  int *NodeSibling;
  
  ptrdiff_t *TreeNodes_offsets;
  ptrdiff_t *TreePoints_offsets;
  ptrdiff_t *TreeNextnode_offsets;
  
  unsigned char *NodeLevel;
  
  point_data *Points;

  int NumPartImported;





 // private, public, protected: ????



  // for some statistics about the number of imported nodes and points
  long long sum_NumForeignNodes;
  long long sum_NumForeignPoints;

  int FirstNonTopLevelNode;

  int EndOfTreePoints;
  int EndOfForeignNodes;

  int ImportedNodeOffset;
  int Ninsert;
  int NextFreeNode;

  MPI_Comm TreeSharedMemComm;
  int TreeSharedMem_ThisTask;
  int TreeSharedMem_NTask;

  int TreeInfoHandle;

  double Buildtime;

  int NumOnFetchStack;
  int MaxOnFetchStack;



  int NumOnWorkStack;
  int MaxOnWorkStack;
  int NewOnWorkStack;
  int AllocWorkStackBaseLow;
  int AllocWorkStackBaseHigh;

  

  
  struct node_count_info
  {
    int count_nodes;
    int count_parts;
  };

  struct node_req
  {
    int foreigntask;
    int foreignnode;
  };


  enum ftype
  {
    FETCH_GRAVTREE,
    FETCH_SPH_DENSITY,
    FETCH_SPH_HYDRO,
    FETCH_SPH_TREETIMESTEP,
  };







 private:
  /** Gives next node in tree walk for the "particle" nodes. Entries 0
         -- MaxPart-1 are the real particles, and the "pseudoparticles" are
           indexed by the node number-MaxNodes. */

  /** Gives previous node in tree walk for the leaf (particle)
      nodes. Entries 0 -- MaxPart-1 are the real particles, and the
      "pseudoparticles" are indexed by the node number-MaxNodes. */
  
  // FIXME remove this function
  static bool compare_ghostrank(const fetch_data &a, const fetch_data &b) { return a.GhostRank < b.GhostRank; }
  
  void tree_get_node_and_task(int i, int &no, int &task);
 
 protected:
  
  void prepare_shared_memory_access(void);
  void cleanup_shared_memory_access(void);
  
  inline sph_particle_data *get_SphPp(int n, unsigned char shmrank)
  {
    return (sph_particle_data *)((char *)TreeSharedMemBaseAddr[shmrank] + TreeSphP_offsets[shmrank]) + n;
  }
  
  void tree_add_to_fetch_stack(node *nop, int nodetoopen, unsigned char
  shmrank);
  int treebuild_construct(void);
  int treebuild_insert_group_of_points(int num, index_data *index_list, int th, unsigned char level, int sibling);
  int create_empty_nodes(int no, int level, int topnode, int bits, int sibling, MyIntPosType x, MyIntPosType y, MyIntPosType z);

  void tree_export_node_threads_by_task_and_node(int task, int nodeindex, int i, thread_data *thread, offset_tuple off = 0);
  
  
  virtual void update_node_recursive(int no, int sib, int mode)            = 0;
  virtual void exchange_topleafdata(void)                                  = 0;
  virtual void report_log_message(void)                                    = 0;
  virtual void fill_in_export_points(point_data *exp_point, int i, int no) = 0;
  
  
  inline node *get_nodep(int no)
  {
    node *nop;

    if(no < MaxPart + D->NTopnodes)
      nop = &TopNodes[no];
    else if(no < MaxPart + MaxNodes)
      nop = &Nodes[no];
    else
      Terminate("illegale node index");

    return nop;
  }

  inline node *get_nodep(int no, unsigned char shmrank)
  {
    node *nop;

    if(no < MaxPart + D->NTopnodes)
      nop = &TopNodes[no];
    else if(no < MaxPart + MaxNodes) /* an internal node */
      {
        node *Nodes_shmrank = (node *)((char *)TreeSharedMemBaseAddr[shmrank] + TreeNodes_offsets[shmrank]);
        nop                 = &Nodes_shmrank[no];
      }
    else if(no >= EndOfTreePoints && no < EndOfForeignNodes) /* an imported tree node */
      {
        node *Foreign_Nodes_shmrank = (node *)((char *)TreeSharedMemBaseAddr[shmrank] + TreeForeign_Nodes_offsets[shmrank]);

        nop = &Foreign_Nodes_shmrank[no - EndOfTreePoints];
      }
    else
      Terminate("illegale node index");

    return nop;
  }

  inline point_data *get_pointsp(int no, unsigned char shmrank)
  {
    return (point_data *)((char *)TreeSharedMemBaseAddr[shmrank] + TreePoints_offsets[shmrank]) + no;
  }
  void tree_add_to_work_stack(int target, int no, unsigned char shmrank, int mintopleafnode)
  {
    if(NumOnWorkStack + NewOnWorkStack >= MaxOnWorkStack)
      {
        Terminate("we shouldn't get here");
        MaxOnWorkStack *= 1.1;
        WorkStack = (workstack_data *)Mem.myrealloc_movable(WorkStack, MaxOnWorkStack * sizeof(workstack_data));
      }

    WorkStack[NumOnWorkStack + NewOnWorkStack].Target         = target;
    WorkStack[NumOnWorkStack + NewOnWorkStack].Node           = no;
    WorkStack[NumOnWorkStack + NewOnWorkStack].ShmRank        = shmrank;
    WorkStack[NumOnWorkStack + NewOnWorkStack].MinTopLeafNode = mintopleafnode;

    NewOnWorkStack++;
  }
  inline foreign_point_data *get_foreignpointsp(int n, unsigned char shmrank)
  {
    return (foreign_point_data *)((char *)TreeSharedMemBaseAddr[shmrank] + TreeForeign_Points_offsets[shmrank]) + n;
  }
  
  void tree_fetch_foreign_nodes(enum ftype fetch_type);
  void tree_initialize_leaf_node_access_info(void);
  
 public:
  tree() /* constructor */
  {
    TopNodes    = NULL;
    Nodes       = NULL;
    NodeIndex   = NULL;
    NodeSibling = NULL;
    NodeLevel   = NULL;
    Points      = NULL;
    Nextnode    = NULL;
    Father      = NULL;
    D           = NULL;
  }
  inline pdata get_Pp(int n, unsigned char shmrank)
  {
    return (pdata)((char *)TreeSharedMemBaseAddr[shmrank] + TreeP_offsets[shmrank]) + n;
  }
  inline subfind_data *get_PSp(int n, unsigned char shmrank)
  {
    return (subfind_data *)((char *)TreeSharedMemBaseAddr[shmrank] + TreePS_offsets[shmrank]) + n;
  }

  /** public functions */
  int treebuild(int ninsert, int *indexlist);
  void treefree(void);
  void treeallocate(int max_partindex, partset *Pptr, domain<partset> *Dptr);
  void treeallocate_share_topnode_addresses(void);
  
  void tree_export_node_threads(int no, int i, thread_data *thread, offset_tuple off = 0);

  inline int *get_nextnodep(unsigned char shmrank)
  {
    return (int *)((char *)TreeSharedMemBaseAddr[shmrank] + TreeNextnode_offsets[shmrank]);
  }
};

#endif
