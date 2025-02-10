
#define MAX_THREADS_PER_BLOCK 512
#define NUM_SM 68 //the number of Streaming Multiprocessors; 15 for Fermi architecture 30 for G280 at the moment of this document
#define NUM_BIN 8 //the number of duplicated frontiers used in BFS_kernel_multi_blk_inGPU
#define EXP 3 // EXP = log(NUM_BIN), assuming NUM_BIN is still power of 2 in the future architecture
	//using EXP and shifting can speed up division operation 
#define MOD_OP 7 // This variable is also related with NUM_BIN; may change in the future architecture;
	//using MOD_OP and "bitwise and" can speed up mod operation
#define INF 2147483647//2^31-1
#define UP_LIMIT 16677216//2^24
#define WHITE 16677217
#define GRAY 16677218
#define GRAY0 16677219
#define GRAY1 16677220
#define BLACK 16677221
#define W_QUEUE_SIZE 400

FILE *fp;

typedef int2 Node;
typedef int2 Edge;

const int h_top = 1;
const int zero = 0;

texture<Node> bfs_ori_graph_node_ref;
texture<Edge> bfs_ori_graph_edge_ref;
texture<Node> bfs_ptb_graph_node_ref;
texture<Edge> bfs_ptb_graph_edge_ref;

volatile __device__ int count = 0;
volatile __device__ int no_of_nodes_vol = 0;
volatile __device__ int stay_vol = 0;

#include "pets_common.h"
#define BFS_GRID_DIM (SM_NUM * 1)

// A group of local queues of node IDs, used by an entire thread block.
// Multiple queues are used to reduce memory contention.
// Thread i uses queue number (i % NUM_BIN).
struct LocalQueues {
	// tail[n] is the index of the first empty array in elems[n]
	int tail[NUM_BIN];

	// Queue elements.
	// The contents of queue n are elems[n][0 .. tail[n] - 1].
	int elems[NUM_BIN][W_QUEUE_SIZE];
	int sharers[NUM_BIN];

	__device__ void reset(int index, dim3 block_dim) {
		tail[index] = 0;		// Queue contains nothing

		// Number of sharers is (threads per block / number of queues)
		// If division is not exact, assign the leftover threads to the first
		// few queues.
		sharers[index] = (block_dim.x >> EXP) +
			(threadIdx.x < (block_dim.x & MOD_OP));
	}

	__device__ void append(int index, int *overflow, int value) {
		// Queue may be accessed concurrently, so
		// use an atomic operation to reserve a queue index.
		int tail_index = atomicAdd(&tail[index], 1);
		if (tail_index >= W_QUEUE_SIZE)
			*overflow = 1;
		else
			elems[index][tail_index] = value;
	}

	__device__ int size_prefix_sum(int (&prefix_q)[NUM_BIN]) {
		prefix_q[0] = 0;
		for(int i = 1; i < NUM_BIN; i++){
			prefix_q[i] = prefix_q[i-1] + tail[i-1];
		}
		return prefix_q[NUM_BIN-1] + tail[NUM_BIN-1];
	}

	__device__ void concatenate(int *dst, int (&prefix_q)[NUM_BIN]) {
		// Thread n processes elems[n % NUM_BIN][n / NUM_BIN, ...]
		int q_i = threadIdx.x & MOD_OP; // w-queue index
		int local_shift = threadIdx.x >> EXP; // shift within a w-queue

		while(local_shift < tail[q_i]){
			dst[prefix_q[q_i] + local_shift] = elems[q_i][local_shift];

			//multiple threads are copying elements at the same time,
			//so we shift by multiple elements for next iteration  
			local_shift += sharers[q_i];
		}
	}
};


__device__ void start_global_barrier(int fold) {
	__syncthreads();

	if(threadIdx.x == 0) {
		atomicAdd((int*)&count, 1);
		while(count < NUM_SM*fold) {
			;
		}
	}
	__syncthreads();
}


__device__ void visit_node(int pid,
	   int index,
	   LocalQueues &local_q,
	   int *overflow,
	   int *g_color,
	   int *g_cost,
	   int gray_shade)
{
	g_color[pid] = BLACK;		// Mark this node as visited
	int cur_cost = g_cost[pid];	// Look up shortest-path distance to this node
	Node cur_node = tex1Dfetch(bfs_ori_graph_node_ref,pid);

	// For each outgoing edge
	for(int i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
		Edge cur_edge = tex1Dfetch(bfs_ori_graph_edge_ref,i);
		int id = cur_edge.x;
		int cost = cur_edge.y;
		cost += cur_cost;
		int orig_cost = atomicMin(&g_cost[id],cost);

		// If this outgoing edge makes a shorter path than any previously
		// discovered path
		if(orig_cost > cost){
			int old_color = atomicExch(&g_color[id],gray_shade);
			if(old_color != gray_shade) {
				//push to the queue
				local_q.append(index, overflow, id);
			}
		}
	}
}


__device__ void ptb_visit_node(int pid,
	   int index,
	   LocalQueues &local_q,
	   int *overflow,
	   int *g_color,
	   int *g_cost,
	   int gray_shade)
{
	g_color[pid] = BLACK;		// Mark this node as visited
	int cur_cost = g_cost[pid];	// Look up shortest-path distance to this node
	Node cur_node = tex1Dfetch(bfs_ptb_graph_node_ref,pid);

	// For each outgoing edge
	for(int i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
		Edge cur_edge = tex1Dfetch(bfs_ptb_graph_edge_ref,i);
		int id = cur_edge.x;
		int cost = cur_edge.y;
		cost += cur_cost;
		int orig_cost = atomicMin(&g_cost[id],cost);

		// If this outgoing edge makes a shorter path than any previously
		// discovered path
		if(orig_cost > cost){
			int old_color = atomicExch(&g_color[id],gray_shade);
			if(old_color != gray_shade) {
				//push to the queue
				local_q.append(index, overflow, id);
			}
		}
	}
}

