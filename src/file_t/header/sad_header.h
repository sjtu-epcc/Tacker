/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

struct image_i16
{
  int width;
  int height;
  short *data;
};

/* Search offsets within 16 pixels of (0,0) */
#define SEARCH_RANGE 16

/* The total search area is 33 pixels square */
#define SEARCH_DIMENSION (2*SEARCH_RANGE+1)

/* The total number of search positions is 33^2 */
#define MAX_POS 1089

/* This is padded to a multiple of 8 when allocating memory */
#define MAX_POS_PADDED 1096

/* VBSME block indices in the SAD array for different 
 * block sizes.  The index is computed from the
 * image size in macroblocks.  Block sizes are (height, width):
 *  1: 16 by 16 pixels, one block per macroblock
 *  2: 8  by 16 pixels, 2  blocks per macroblock
 *  3: 16 by 8  pixels, 2  blocks per macroblock
 *  4: 8  by 8  pixels, 4  blocks per macroblock
 *  5: 4  by 8  pixels, 8  blocks per macroblock
 *  6: 8  by 4  pixels, 8  blocks per macroblock
 *  7: 4  by 4  pixels, 16 blocks per macroblock
 */
#define SAD_TYPE_1_IX(image_size) 0
#define SAD_TYPE_2_IX(image_size) ((image_size)*MAX_POS_PADDED)
#define SAD_TYPE_3_IX(image_size) ((image_size)*(3*MAX_POS_PADDED))
#define SAD_TYPE_4_IX(image_size) ((image_size)*(5*MAX_POS_PADDED))
#define SAD_TYPE_5_IX(image_size) ((image_size)*(9*MAX_POS_PADDED))
#define SAD_TYPE_6_IX(image_size) ((image_size)*(17*MAX_POS_PADDED))
#define SAD_TYPE_7_IX(image_size) ((image_size)*(25*MAX_POS_PADDED))

#define SAD_TYPE_IX(n, image_size) \
  ((n == 1) ? SAD_TYPE_1_IX(image_size) : \
   ((n == 2) ? SAD_TYPE_2_IX(image_size) : \
    ((n == 3) ? SAD_TYPE_3_IX(image_size) : \
     ((n == 4) ? SAD_TYPE_4_IX(image_size) : \
      ((n == 5) ? SAD_TYPE_5_IX(image_size) : \
       ((n == 6) ? SAD_TYPE_6_IX(image_size) : \
        (SAD_TYPE_7_IX(image_size) \
	 )))))))

#define SAD_TYPE_1_CT 1
#define SAD_TYPE_2_CT 2
#define SAD_TYPE_3_CT 2
#define SAD_TYPE_4_CT 4
#define SAD_TYPE_5_CT 8
#define SAD_TYPE_6_CT 8
#define SAD_TYPE_7_CT 16

#define SAD_TYPE_CT(n) \
  ((n == 1) ? SAD_TYPE_1_CT : \
   ((n == 2) ? SAD_TYPE_2_CT : \
    ((n == 3) ? SAD_TYPE_3_CT : \
     ((n == 4) ? SAD_TYPE_4_CT : \
      ((n == 5) ? SAD_TYPE_5_CT : \
       ((n == 6) ? SAD_TYPE_6_CT : \
        (SAD_TYPE_7_CT \
	 )))))))




/* Integer ceiling division.  This computes ceil(x / y) */
#define CEIL(x,y) (((x) + ((y) - 1)) / (y))

/* Fast multiplication by 33 */
#define TIMES_DIM_POS(x) (((x) << 5) + (x))

/* Amount of dynamically allocated local storage
 * measured in bytes, 2-byte words, and 8-byte words */
#define SAD_LOC_SIZE_ELEMS (THREADS_W * THREADS_H * MAX_POS_PADDED)
#define SAD_LOC_SIZE_BYTES (SAD_LOC_SIZE_ELEMS * sizeof(unsigned short))
#define SAD_LOC_SIZE_8B (SAD_LOC_SIZE_BYTES / sizeof(vec8b))

/* The search position index space is distributed across threads
 * and across time. */
/* This many search positions are calculated by each thread.
 * Note: the optimized kernel requires that this number is
 * divisible by 3. */
#define POS_PER_THREAD 18

/* The width and height (in number of 4x4 blocks) of a tile from the
 * current frame that is computed in a single thread block. */
#define THREADS_W 1
#define THREADS_H 1

// #define TIMES_THREADS_W(x) (((x) << 1) + (x))
#define TIMES_THREADS_W(x) ((x) * THREADS_W)

/* This structure is used for vector load/store operations. */
struct vec8b {
  int fst;
  int snd;
} __align__(8);

typedef struct vec8b vec8b;

/* A function to get a reference to the "ref" texture, because sharing
 * of textures between files isn't really supported. */
texture<unsigned short, 2, cudaReadModeElementType> &get_ref(void);
texture<unsigned short, 2, cudaReadModeElementType> &get_ref_2(void);

/* Macros to access temporary frame storage in shared memory */
#define FRAME_GET(n, x, y) \
  (frame_loc[((n) << 4) + ((y) << 2) + (x)])
#define FRAME_PUT_1(n, x, value) \
  (frame_loc[((n) << 4) + (x)] = value)

/* Macros to access temporary SAD storage in shared memory */
#define SAD_LOC_GET(blocknum, pos) \
  (sad_loc[(blocknum) * MAX_POS_PADDED + (pos)])
#define SAD_LOC_PUT(blocknum, pos, value) \
  (sad_loc[(blocknum) * MAX_POS_PADDED + (pos)] = (value))

/* When reading from this array, we use an "index" rather than a
   search position.  Also, the number of array elements is divided by
   four relative to SAD_LOC_GET() since this is an array of 8byte
   data, while SAD_LOC_GET() sees an array of 2byte data. */
#define SAD_LOC_8B_GET(blocknum, ix) \
  (sad_loc_8b[(blocknum) * (MAX_POS_PADDED/4) + (ix)])

/* The size of one row of sad_loc_8b.  This is the group of elements
 * holding SADs for all search positions for one 4x4 block. */
#define SAD_LOC_8B_ROW_SIZE (MAX_POS_PADDED/4)

/* The presence of this preprocessor variable controls which
 * of two means of computing the current search position is used. */
#define SEARCHPOS_RECURRENCE

/* A local copy of the current 4x4 block */
__shared__ unsigned short frame_loc[THREADS_W * THREADS_H * 16];

/* The part of the reference image that is in the search range */
texture<unsigned short, 2, cudaReadModeElementType> ref;
texture<unsigned short, 2, cudaReadModeElementType> ref_2;