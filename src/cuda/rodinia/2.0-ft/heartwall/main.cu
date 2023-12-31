//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	DEFINE / INCLUDE
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

//======================================================================================================================================================
//	LIBRARIES
//======================================================================================================================================================

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <avilib.h>
#include <avimod.h>
#include <cuda.h>

//======================================================================================================================================================
//	STRUCTURES, GLOBAL STRUCTURE VARIABLES
//======================================================================================================================================================

#include "define.c"

params_common_change* common_change_ptr;
__constant__ params_common_change d_common_change;

//params_common common;
params_common* common_ptr;
__constant__ params_common d_common;

//params_unique unique[ALL_POINTS];								// cannot determine size dynamically so choose more than usually needed
params_unique* unique_ptr;
__constant__ params_unique d_unique[ALL_POINTS];

//======================================================================================================================================================
// KERNEL CODE
//======================================================================================================================================================

#include "kernel.cu"

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	MAIN FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

int main(int argc, char *argv []){

	//======================================================================================================================================================
	//	VARIABLES
	//======================================================================================================================================================

	// CUDA kernel execution parameters
	dim3 threads;
	dim3 blocks;

	// counter
	int i;
	int frames_processed;

	// frames
	char* video_file_name;
	avi_t* frames;
	fp* frame;

	//======================================================================================================================================================
	// 	FRAME
	//======================================================================================================================================================

	if(argc!=4){
		printf("ERROR: usage: heartwall <inputfile> <num of frames> <goldfile>\n");
		exit(1);
	}
	const char* goldfile = argv[3];	
	// open movie file
 	video_file_name = argv[1];
	frames = (avi_t*)AVI_open_input_file(video_file_name, 1);														// added casting
	if (frames == NULL)  {
		   AVI_print_error((char *) "Error with AVI_open_input_file");
		   return -1;
	}

	// common
    cudaMallocHost((void**) &common_ptr, sizeof(params_common));

	common_ptr->no_frames = AVI_video_frames(frames);
	common_ptr->frame_rows = AVI_video_height(frames);
	common_ptr->frame_cols = AVI_video_width(frames);
	common_ptr->frame_elem = common_ptr->frame_rows * common_ptr->frame_cols;
	common_ptr->frame_mem = sizeof(fp) * common_ptr->frame_elem;

	// pointers
    cudaMallocHost((void**) &common_change_ptr, sizeof(params_common_change));
	cudaMalloc((void **)&common_change_ptr->d_frame, common_ptr->frame_mem);

	//======================================================================================================================================================
	// 	CHECK INPUT ARGUMENTS
	//======================================================================================================================================================
	
	frames_processed = atoi(argv[2]);
		if(frames_processed<0 || frames_processed>common_ptr->no_frames){
			printf("ERROR: %d is an incorrect number of frames specified, select in the range of 0-%d\n", frames_processed, common_ptr->no_frames);
			return 0;
	}
	

	//======================================================================================================================================================
	//	HARDCODED INPUTS FROM MATLAB
	//======================================================================================================================================================

	//====================================================================================================
	//	CONSTANTS
	//====================================================================================================

	common_ptr->sSize = 40;
	common_ptr->tSize = 25;
	common_ptr->maxMove = 10;
	common_ptr->alpha = 0.87;

	//====================================================================================================
	//	ENDO POINTS
	//====================================================================================================

	common_ptr->endoPoints = ENDO_POINTS;
	common_ptr->endo_mem = sizeof(int) * common_ptr->endoPoints;

	//common_ptr->endoRow = (int *)malloc(common_ptr->endo_mem);
    cudaMallocHost((void**) &common_ptr->endoRow, common_ptr->endo_mem);
	common_ptr->endoRow[ 0] = 369;
	common_ptr->endoRow[ 1] = 400;
	common_ptr->endoRow[ 2] = 429;
	common_ptr->endoRow[ 3] = 452;
	common_ptr->endoRow[ 4] = 476;
	common_ptr->endoRow[ 5] = 486;
	common_ptr->endoRow[ 6] = 479;
	common_ptr->endoRow[ 7] = 458;
	common_ptr->endoRow[ 8] = 433;
	common_ptr->endoRow[ 9] = 404;
	common_ptr->endoRow[10] = 374;
	common_ptr->endoRow[11] = 346;
	common_ptr->endoRow[12] = 318;
	common_ptr->endoRow[13] = 294;
	common_ptr->endoRow[14] = 277;
	common_ptr->endoRow[15] = 269;
	common_ptr->endoRow[16] = 275;
	common_ptr->endoRow[17] = 287;
	common_ptr->endoRow[18] = 311;
	common_ptr->endoRow[19] = 339;
	cudaMalloc((void **)&common_ptr->d_endoRow, common_ptr->endo_mem);
	cudaMemcpy(common_ptr->d_endoRow, common_ptr->endoRow, common_ptr->endo_mem, cudaMemcpyHostToDevice);

	//common_ptr->endoCol = (int *)malloc(common_ptr->endo_mem);
    cudaMallocHost((void**) &common_ptr->endoCol, common_ptr->endo_mem);
	common_ptr->endoCol[ 0] = 408;
	common_ptr->endoCol[ 1] = 406;
	common_ptr->endoCol[ 2] = 397;
	common_ptr->endoCol[ 3] = 383;
	common_ptr->endoCol[ 4] = 354;
	common_ptr->endoCol[ 5] = 322;
	common_ptr->endoCol[ 6] = 294;
	common_ptr->endoCol[ 7] = 270;
	common_ptr->endoCol[ 8] = 250;
	common_ptr->endoCol[ 9] = 237;
	common_ptr->endoCol[10] = 235;
	common_ptr->endoCol[11] = 241;
	common_ptr->endoCol[12] = 254;
	common_ptr->endoCol[13] = 273;
	common_ptr->endoCol[14] = 300;
	common_ptr->endoCol[15] = 328;
	common_ptr->endoCol[16] = 356;
	common_ptr->endoCol[17] = 383;
	common_ptr->endoCol[18] = 401;
	common_ptr->endoCol[19] = 411;
	cudaMalloc((void **)&common_ptr->d_endoCol, common_ptr->endo_mem);
	cudaMemcpy(common_ptr->d_endoCol, common_ptr->endoCol, common_ptr->endo_mem, cudaMemcpyHostToDevice);

	//common_ptr->tEndoRowLoc = (int *)calloc(common_ptr->endo_mem * common_ptr->no_frames, 1);
    cudaMallocHost((void **) &common_ptr->tEndoRowLoc, common_ptr->endo_mem * common_ptr->no_frames);
	cudaMalloc((void **)&common_ptr->d_tEndoRowLoc, common_ptr->endo_mem * common_ptr->no_frames);
	cudaMemset((void *)common_ptr->d_tEndoRowLoc, 0, common_ptr->endo_mem * common_ptr->no_frames);

	//common_ptr->tEndoColLoc = (int *)calloc(common_ptr->endo_mem * common_ptr->no_frames, 1);
    cudaMallocHost((void **) &common_ptr->tEndoColLoc, common_ptr->endo_mem * common_ptr->no_frames);
	cudaMalloc((void **)&common_ptr->d_tEndoColLoc, common_ptr->endo_mem * common_ptr->no_frames);
	cudaMemset((void *)common_ptr->d_tEndoColLoc, 0, common_ptr->endo_mem * common_ptr->no_frames);

	//====================================================================================================
	//	EPI POINTS
	//====================================================================================================

	common_ptr->epiPoints = EPI_POINTS;
	common_ptr->epi_mem = sizeof(int) * common_ptr->epiPoints;

	//common_ptr->epiRow = (int *)malloc(common_ptr->epi_mem);
    cudaMallocHost((void**) &common_ptr->epiRow, common_ptr->epi_mem);
	common_ptr->epiRow[ 0] = 390;
	common_ptr->epiRow[ 1] = 419;
	common_ptr->epiRow[ 2] = 448;
	common_ptr->epiRow[ 3] = 474;
	common_ptr->epiRow[ 4] = 501;
	common_ptr->epiRow[ 5] = 519;
	common_ptr->epiRow[ 6] = 535;
	common_ptr->epiRow[ 7] = 542;
	common_ptr->epiRow[ 8] = 543;
	common_ptr->epiRow[ 9] = 538;
	common_ptr->epiRow[10] = 528;
	common_ptr->epiRow[11] = 511;
	common_ptr->epiRow[12] = 491;
	common_ptr->epiRow[13] = 466;
	common_ptr->epiRow[14] = 438;
	common_ptr->epiRow[15] = 406;
	common_ptr->epiRow[16] = 376;
	common_ptr->epiRow[17] = 347;
	common_ptr->epiRow[18] = 318;
	common_ptr->epiRow[19] = 291;
	common_ptr->epiRow[20] = 275;
	common_ptr->epiRow[21] = 259;
	common_ptr->epiRow[22] = 256;
	common_ptr->epiRow[23] = 252;
	common_ptr->epiRow[24] = 252;
	common_ptr->epiRow[25] = 257;
	common_ptr->epiRow[26] = 266;
	common_ptr->epiRow[27] = 283;
	common_ptr->epiRow[28] = 305;
	common_ptr->epiRow[29] = 331;
	common_ptr->epiRow[30] = 360;
	cudaMalloc((void **)&common_ptr->d_epiRow, common_ptr->epi_mem);
	cudaMemcpy(common_ptr->d_epiRow, common_ptr->epiRow, common_ptr->epi_mem, cudaMemcpyHostToDevice);

	//common_ptr->epiCol = (int *)malloc(common_ptr->epi_mem);
    cudaMallocHost((void**) &common_ptr->epiCol, common_ptr->epi_mem);
	common_ptr->epiCol[ 0] = 457;
	common_ptr->epiCol[ 1] = 454;
	common_ptr->epiCol[ 2] = 446;
	common_ptr->epiCol[ 3] = 431;
	common_ptr->epiCol[ 4] = 411;
	common_ptr->epiCol[ 5] = 388;
	common_ptr->epiCol[ 6] = 361;
	common_ptr->epiCol[ 7] = 331;
	common_ptr->epiCol[ 8] = 301;
	common_ptr->epiCol[ 9] = 273;
	common_ptr->epiCol[10] = 243;
	common_ptr->epiCol[11] = 218;
	common_ptr->epiCol[12] = 196;
	common_ptr->epiCol[13] = 178;
	common_ptr->epiCol[14] = 166;
	common_ptr->epiCol[15] = 157;
	common_ptr->epiCol[16] = 155;
	common_ptr->epiCol[17] = 165;
	common_ptr->epiCol[18] = 177;
	common_ptr->epiCol[19] = 197;
	common_ptr->epiCol[20] = 218;
	common_ptr->epiCol[21] = 248;
	common_ptr->epiCol[22] = 276;
	common_ptr->epiCol[23] = 304;
	common_ptr->epiCol[24] = 333;
	common_ptr->epiCol[25] = 361;
	common_ptr->epiCol[26] = 391;
	common_ptr->epiCol[27] = 415;
	common_ptr->epiCol[28] = 434;
	common_ptr->epiCol[29] = 448;
	common_ptr->epiCol[30] = 455;
	cudaMalloc((void **)&common_ptr->d_epiCol, common_ptr->epi_mem);
	cudaMemcpy(common_ptr->d_epiCol, common_ptr->epiCol, common_ptr->epi_mem, cudaMemcpyHostToDevice);

	//common_ptr->tEpiRowLoc = (int *)calloc(common_ptr->epi_mem * common_ptr->no_frames, 1);
    cudaMallocHost((void **) &common_ptr->tEpiRowLoc, common_ptr->epi_mem * common_ptr->no_frames);
	cudaMalloc((void **)&common_ptr->d_tEpiRowLoc, common_ptr->epi_mem * common_ptr->no_frames);
	cudaMemset((void *)common_ptr->d_tEpiRowLoc, 0, common_ptr->epi_mem * common_ptr->no_frames);

	//common_ptr->tEpiColLoc = (int *)calloc(common_ptr->epi_mem * common_ptr->no_frames, 1);
    cudaMallocHost((void **) &common_ptr->tEpiColLoc, common_ptr->epi_mem * common_ptr->no_frames);
	cudaMalloc((void **)&common_ptr->d_tEpiColLoc, common_ptr->epi_mem * common_ptr->no_frames);
	cudaMemset((void *)common_ptr->d_tEpiColLoc, 0, common_ptr->epi_mem * common_ptr->no_frames);

	//====================================================================================================
	//	ALL POINTS
	//====================================================================================================

	common_ptr->allPoints = ALL_POINTS;

	//======================================================================================================================================================
	// 	TEMPLATE SIZES
	//======================================================================================================================================================

	// common
	common_ptr->in_rows = common_ptr->tSize + 1 + common_ptr->tSize;
	common_ptr->in_cols = common_ptr->in_rows;
	common_ptr->in_elem = common_ptr->in_rows * common_ptr->in_cols;
	common_ptr->in_mem = sizeof(fp) * common_ptr->in_elem;

	//======================================================================================================================================================
	// 	CREATE ARRAY OF TEMPLATES FOR ALL POINTS
	//======================================================================================================================================================

	// common
	cudaMalloc((void **)&common_ptr->d_endoT, common_ptr->in_mem * common_ptr->endoPoints);
	cudaMalloc((void **)&common_ptr->d_epiT, common_ptr->in_mem * common_ptr->epiPoints);

	//======================================================================================================================================================
	//	SPECIFIC TO ENDO OR EPI TO BE SET HERE
	//======================================================================================================================================================

    cudaMallocHost((void**) &unique_ptr, ALL_POINTS * sizeof(params_unique));

	for(i=0; i<common_ptr->endoPoints; i++){
		unique_ptr[i].point_no = i;
		unique_ptr[i].d_Row = common_ptr->d_endoRow;
		unique_ptr[i].d_Col = common_ptr->d_endoCol;
		unique_ptr[i].d_tRowLoc = common_ptr->d_tEndoRowLoc;
		unique_ptr[i].d_tColLoc = common_ptr->d_tEndoColLoc;
		unique_ptr[i].d_T = common_ptr->d_endoT;
	}
	for(i=common_ptr->endoPoints; i<common_ptr->allPoints; i++){
		unique_ptr[i].point_no = i-common_ptr->endoPoints;
		unique_ptr[i].d_Row = common_ptr->d_epiRow;
		unique_ptr[i].d_Col = common_ptr->d_epiCol;
		unique_ptr[i].d_tRowLoc = common_ptr->d_tEpiRowLoc;
		unique_ptr[i].d_tColLoc = common_ptr->d_tEpiColLoc;
		unique_ptr[i].d_T = common_ptr->d_epiT;
	}

	//======================================================================================================================================================
	// 	RIGHT TEMPLATE 	FROM 	TEMPLATE ARRAY
	//======================================================================================================================================================

	// pointers
	for(i=0; i<common_ptr->allPoints; i++){
		unique_ptr[i].in_pointer = unique_ptr[i].point_no * common_ptr->in_elem;
	}

	//======================================================================================================================================================
	// 	AREA AROUND POINT		FROM	FRAME
	//======================================================================================================================================================

	// common
	common_ptr->in2_rows = 2 * common_ptr->sSize + 1;
	common_ptr->in2_cols = 2 * common_ptr->sSize + 1;
	common_ptr->in2_elem = common_ptr->in2_rows * common_ptr->in2_cols;
	common_ptr->in2_mem = sizeof(float) * common_ptr->in2_elem;

	// pointers
	for(i=0; i<common_ptr->allPoints; i++){
		cudaMalloc((void **)&unique_ptr[i].d_in2, common_ptr->in2_mem);
	}

	//======================================================================================================================================================
	// 	CONVOLUTION
	//======================================================================================================================================================

	// common
	common_ptr->conv_rows = common_ptr->in_rows + common_ptr->in2_rows - 1;												// number of rows in I
	common_ptr->conv_cols = common_ptr->in_cols + common_ptr->in2_cols - 1;												// number of columns in I
	common_ptr->conv_elem = common_ptr->conv_rows * common_ptr->conv_cols;												// number of elements
	common_ptr->conv_mem = sizeof(float) * common_ptr->conv_elem;
	common_ptr->ioffset = 0;
	common_ptr->joffset = 0;

	// pointers
	for(i=0; i<common_ptr->allPoints; i++){
		cudaMalloc((void **)&unique_ptr[i].d_conv, common_ptr->conv_mem);
	}

	//======================================================================================================================================================
	// 	CUMULATIVE SUM
	//======================================================================================================================================================

	//====================================================================================================
	// 	PADDING OF ARRAY, VERTICAL CUMULATIVE SUM
	//====================================================================================================

	// common
	common_ptr->in2_pad_add_rows = common_ptr->in_rows;
	common_ptr->in2_pad_add_cols = common_ptr->in_cols;

	common_ptr->in2_pad_cumv_rows = common_ptr->in2_rows + 2*common_ptr->in2_pad_add_rows;
	common_ptr->in2_pad_cumv_cols = common_ptr->in2_cols + 2*common_ptr->in2_pad_add_cols;
	common_ptr->in2_pad_cumv_elem = common_ptr->in2_pad_cumv_rows * common_ptr->in2_pad_cumv_cols;
	common_ptr->in2_pad_cumv_mem = sizeof(float) * common_ptr->in2_pad_cumv_elem;

	// pointers
	for(i=0; i<common_ptr->allPoints; i++){
		cudaMalloc((void **)&unique_ptr[i].d_in2_pad_cumv, common_ptr->in2_pad_cumv_mem);
	}

	//====================================================================================================
	// 	SELECTION
	//====================================================================================================

	// common
	common_ptr->in2_pad_cumv_sel_rowlow = 1 + common_ptr->in_rows;													// (1 to n+1)
	common_ptr->in2_pad_cumv_sel_rowhig = common_ptr->in2_pad_cumv_rows - 1;
	common_ptr->in2_pad_cumv_sel_collow = 1;
	common_ptr->in2_pad_cumv_sel_colhig = common_ptr->in2_pad_cumv_cols;
	common_ptr->in2_pad_cumv_sel_rows = common_ptr->in2_pad_cumv_sel_rowhig - common_ptr->in2_pad_cumv_sel_rowlow + 1;
	common_ptr->in2_pad_cumv_sel_cols = common_ptr->in2_pad_cumv_sel_colhig - common_ptr->in2_pad_cumv_sel_collow + 1;
	common_ptr->in2_pad_cumv_sel_elem = common_ptr->in2_pad_cumv_sel_rows * common_ptr->in2_pad_cumv_sel_cols;
	common_ptr->in2_pad_cumv_sel_mem = sizeof(float) * common_ptr->in2_pad_cumv_sel_elem;

	// pointers
	for(i=0; i<common_ptr->allPoints; i++){
		cudaMalloc((void **)&unique_ptr[i].d_in2_pad_cumv_sel, common_ptr->in2_pad_cumv_sel_mem);
	}

	//====================================================================================================
	// 	SELECTION	2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
	//====================================================================================================

	// common
	common_ptr->in2_pad_cumv_sel2_rowlow = 1;
	common_ptr->in2_pad_cumv_sel2_rowhig = common_ptr->in2_pad_cumv_rows - common_ptr->in_rows - 1;
	common_ptr->in2_pad_cumv_sel2_collow = 1;
	common_ptr->in2_pad_cumv_sel2_colhig = common_ptr->in2_pad_cumv_cols;
	common_ptr->in2_sub_cumh_rows = common_ptr->in2_pad_cumv_sel2_rowhig - common_ptr->in2_pad_cumv_sel2_rowlow + 1;
	common_ptr->in2_sub_cumh_cols = common_ptr->in2_pad_cumv_sel2_colhig - common_ptr->in2_pad_cumv_sel2_collow + 1;
	common_ptr->in2_sub_cumh_elem = common_ptr->in2_sub_cumh_rows * common_ptr->in2_sub_cumh_cols;
	common_ptr->in2_sub_cumh_mem = sizeof(float) * common_ptr->in2_sub_cumh_elem;

	// pointers
	for(i=0; i<common_ptr->allPoints; i++){
		cudaMalloc((void **)&unique_ptr[i].d_in2_sub_cumh, common_ptr->in2_sub_cumh_mem);
	}

	//====================================================================================================
	// 	SELECTION
	//====================================================================================================

	// common
	common_ptr->in2_sub_cumh_sel_rowlow = 1;
	common_ptr->in2_sub_cumh_sel_rowhig = common_ptr->in2_sub_cumh_rows;
	common_ptr->in2_sub_cumh_sel_collow = 1 + common_ptr->in_cols;
	common_ptr->in2_sub_cumh_sel_colhig = common_ptr->in2_sub_cumh_cols - 1;
	common_ptr->in2_sub_cumh_sel_rows = common_ptr->in2_sub_cumh_sel_rowhig - common_ptr->in2_sub_cumh_sel_rowlow + 1;
	common_ptr->in2_sub_cumh_sel_cols = common_ptr->in2_sub_cumh_sel_colhig - common_ptr->in2_sub_cumh_sel_collow + 1;
	common_ptr->in2_sub_cumh_sel_elem = common_ptr->in2_sub_cumh_sel_rows * common_ptr->in2_sub_cumh_sel_cols;
	common_ptr->in2_sub_cumh_sel_mem = sizeof(float) * common_ptr->in2_sub_cumh_sel_elem;

	// pointers
	for(i=0; i<common_ptr->allPoints; i++){
		cudaMalloc((void **)&unique_ptr[i].d_in2_sub_cumh_sel, common_ptr->in2_sub_cumh_sel_mem);
	}

	//====================================================================================================
	//	SELECTION 2, SUBTRACTION
	//====================================================================================================

	// common
	common_ptr->in2_sub_cumh_sel2_rowlow = 1;
	common_ptr->in2_sub_cumh_sel2_rowhig = common_ptr->in2_sub_cumh_rows;
	common_ptr->in2_sub_cumh_sel2_collow = 1;
	common_ptr->in2_sub_cumh_sel2_colhig = common_ptr->in2_sub_cumh_cols - common_ptr->in_cols - 1;
	common_ptr->in2_sub2_rows = common_ptr->in2_sub_cumh_sel2_rowhig - common_ptr->in2_sub_cumh_sel2_rowlow + 1;
	common_ptr->in2_sub2_cols = common_ptr->in2_sub_cumh_sel2_colhig - common_ptr->in2_sub_cumh_sel2_collow + 1;
	common_ptr->in2_sub2_elem = common_ptr->in2_sub2_rows * common_ptr->in2_sub2_cols;
	common_ptr->in2_sub2_mem = sizeof(float) * common_ptr->in2_sub2_elem;

	// pointers
	for(i=0; i<common_ptr->allPoints; i++){
		cudaMalloc((void **)&unique_ptr[i].d_in2_sub2, common_ptr->in2_sub2_mem);
	}

	//======================================================================================================================================================
	//	CUMULATIVE SUM 2
	//======================================================================================================================================================

	//====================================================================================================
	//	MULTIPLICATION
	//====================================================================================================

	// common
	common_ptr->in2_sqr_rows = common_ptr->in2_rows;
	common_ptr->in2_sqr_cols = common_ptr->in2_cols;
	common_ptr->in2_sqr_elem = common_ptr->in2_elem;
	common_ptr->in2_sqr_mem = common_ptr->in2_mem;

	// pointers
	for(i=0; i<common_ptr->allPoints; i++){
		cudaMalloc((void **)&unique_ptr[i].d_in2_sqr, common_ptr->in2_sqr_mem);
	}

	//====================================================================================================
	//	SELECTION 2, SUBTRACTION
	//====================================================================================================

	// common
	common_ptr->in2_sqr_sub2_rows = common_ptr->in2_sub2_rows;
	common_ptr->in2_sqr_sub2_cols = common_ptr->in2_sub2_cols;
	common_ptr->in2_sqr_sub2_elem = common_ptr->in2_sub2_elem;
	common_ptr->in2_sqr_sub2_mem = common_ptr->in2_sub2_mem;

	// pointers
	for(i=0; i<common_ptr->allPoints; i++){
		cudaMalloc((void **)&unique_ptr[i].d_in2_sqr_sub2, common_ptr->in2_sqr_sub2_mem);
	}

	//======================================================================================================================================================
	//	FINAL
	//======================================================================================================================================================

	// common
	common_ptr->in_sqr_rows = common_ptr->in_rows;
	common_ptr->in_sqr_cols = common_ptr->in_cols;
	common_ptr->in_sqr_elem = common_ptr->in_elem;
	common_ptr->in_sqr_mem = common_ptr->in_mem;

	// pointers
	for(i=0; i<common_ptr->allPoints; i++){
		cudaMalloc((void **)&unique_ptr[i].d_in_sqr, common_ptr->in_sqr_mem);
	}

	//======================================================================================================================================================
	//	TEMPLATE MASK CREATE
	//======================================================================================================================================================

	// common
	common_ptr->tMask_rows = common_ptr->in_rows + (common_ptr->sSize+1+common_ptr->sSize) - 1;
	common_ptr->tMask_cols = common_ptr->tMask_rows;
	common_ptr->tMask_elem = common_ptr->tMask_rows * common_ptr->tMask_cols;
	common_ptr->tMask_mem = sizeof(float) * common_ptr->tMask_elem;

	// pointers
	for(i=0; i<common_ptr->allPoints; i++){
		cudaMalloc((void **)&unique_ptr[i].d_tMask, common_ptr->tMask_mem);
	}

	//======================================================================================================================================================
	//	POINT MASK INITIALIZE
	//======================================================================================================================================================

	// common
	common_ptr->mask_rows = common_ptr->maxMove;
	common_ptr->mask_cols = common_ptr->mask_rows;
	common_ptr->mask_elem = common_ptr->mask_rows * common_ptr->mask_cols;
	common_ptr->mask_mem = sizeof(float) * common_ptr->mask_elem;

	//======================================================================================================================================================
	//	MASK CONVOLUTION
	//======================================================================================================================================================

	// common
	common_ptr->mask_conv_rows = common_ptr->tMask_rows;												// number of rows in I
	common_ptr->mask_conv_cols = common_ptr->tMask_cols;												// number of columns in I
	common_ptr->mask_conv_elem = common_ptr->mask_conv_rows * common_ptr->mask_conv_cols;												// number of elements
	common_ptr->mask_conv_mem = sizeof(float) * common_ptr->mask_conv_elem;
	common_ptr->mask_conv_ioffset = (common_ptr->mask_rows-1)/2;
	if((common_ptr->mask_rows-1) % 2 > 0.5){
		common_ptr->mask_conv_ioffset = common_ptr->mask_conv_ioffset + 1;
	}
	common_ptr->mask_conv_joffset = (common_ptr->mask_cols-1)/2;
	if((common_ptr->mask_cols-1) % 2 > 0.5){
		common_ptr->mask_conv_joffset = common_ptr->mask_conv_joffset + 1;
	}

	// pointers
	for(i=0; i<common_ptr->allPoints; i++){
		cudaMalloc((void **)&unique_ptr[i].d_mask_conv, common_ptr->mask_conv_mem);
	}

	//======================================================================================================================================================
	//	KERNEL
	//======================================================================================================================================================

	//====================================================================================================
	//	THREAD BLOCK
	//====================================================================================================

	// All kernels operations within kernel use same max size of threads. Size of block size is set to the size appropriate for max size operation (on padded matrix). Other use subsets of that.
	threads.x = NUMBER_THREADS;											// define the number of threads in the block
	threads.y = 1;
	blocks.x = common_ptr->allPoints;							// define the number of blocks in the grid
	blocks.y = 1;

	//====================================================================================================
	//	COPY ARGUMENTS
	//====================================================================================================

	cudaMemcpyToSymbol(d_common, common_ptr, sizeof(params_common));
	cudaMemcpyToSymbol(d_unique, unique_ptr, sizeof(params_unique)*ALL_POINTS);

	//====================================================================================================
	//	PRINT FRAME PROGRESS START
	//====================================================================================================

	printf("frame progress: ");
	fflush(NULL);

	//====================================================================================================
	//	LAUNCH
	//====================================================================================================

	for(common_change_ptr->frame_no=0; common_change_ptr->frame_no<frames_processed; common_change_ptr->frame_no++){

		// Extract a cropped version of the first frame from the video file
		frame = get_frame(	frames,						// pointer to video file
										common_change_ptr->frame_no,				// number of frame that needs to be returned
										0,								// cropped?
										0,								// scaled?
										1);							// converted

		// copy frame to GPU memory
		cudaMemcpy(common_change_ptr->d_frame, frame, common_ptr->frame_mem, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(d_common_change, common_change_ptr, sizeof(params_common_change));

		// launch GPU kernel
		kernel<<<blocks, threads>>>();

		// free frame after each loop iteration, since AVI library allocates memory for every frame fetched
		cudaFreeHost((void*) frame);

		// print frame progress
		printf("%d ", common_change_ptr->frame_no);
		fflush(NULL);

	}

	//====================================================================================================
	//	PRINT FRAME PROGRESS END
	//====================================================================================================

	printf("\n");
	fflush(NULL);

	//====================================================================================================
	//	OUTPUT
	//====================================================================================================

	cudaMemcpy(common_ptr->tEndoRowLoc, common_ptr->d_tEndoRowLoc, common_ptr->endo_mem * common_ptr->no_frames, cudaMemcpyDeviceToHost);
	cudaMemcpy(common_ptr->tEndoColLoc, common_ptr->d_tEndoColLoc, common_ptr->endo_mem * common_ptr->no_frames, cudaMemcpyDeviceToHost);

	cudaMemcpy(common_ptr->tEpiRowLoc, common_ptr->d_tEpiRowLoc, common_ptr->epi_mem * common_ptr->no_frames, cudaMemcpyDeviceToHost);
	cudaMemcpy(common_ptr->tEpiColLoc, common_ptr->d_tEpiColLoc, common_ptr->epi_mem * common_ptr->no_frames, cudaMemcpyDeviceToHost);
	FILE *ofile = fopen("result.txt", "w");
   for (int x = 0; x < common_ptr->endo_mem * common_ptr->no_frames / sizeof(int); x++) {
      printf("common.tEndoRowLoc[%d] = %d\n", x, common_ptr->tEndoRowLoc[x]); 
      printf("common.tEndoColLoc[%d] = %d\n", x, common_ptr->tEndoColLoc[x]); 
      fprintf(ofile, "common.tEndoRowLoc[%d] = %d\n", x, common_ptr->tEndoRowLoc[x]); 
      fprintf(ofile, "common.tEndoColLoc[%d] = %d\n", x, common_ptr->tEndoColLoc[x]); 
   }

   for (int x = 0; x < common_ptr->epi_mem * common_ptr->no_frames / sizeof(int); x++) {
      printf("common.tEpiRowLoc[%d] = %d\n", x, common_ptr->tEpiRowLoc[x]); 
      printf("common.tEpiColLoc[%d] = %d\n", x, common_ptr->tEpiColLoc[x]); 
      fprintf(ofile, "common.tEpiRowLoc[%d] = %d\n", x, common_ptr->tEpiRowLoc[x]); 
      fprintf(ofile, "common.tEpiColLoc[%d] = %d\n", x, common_ptr->tEpiColLoc[x]); 
   }
	fclose(ofile);
	if(goldfile){
		FILE *gold = fopen(goldfile, "r");
		FILE *result = fopen("result.txt", "r");
		int result_error=0;
		while(!feof(gold)&&!feof(result)){
			if (fgetc(gold)!=fgetc(result)) {
				result_error = 1;
				break;
			}
		}
		if((feof(gold)^feof(result)) | result_error) {
			printf("\nFAILED\n");
		} else {
			printf("\nPASSED\n");
		}

		fclose(gold);
		fclose(result);
	}

	//======================================================================================================================================================
	//	DEALLOCATION
	//======================================================================================================================================================

	//====================================================================================================
	//	COMMON
	//====================================================================================================

	// frame
	cudaFree(common_change_ptr->d_frame);

	// endo points
	cudaFreeHost((void*) common_ptr->endoRow);
	cudaFreeHost((void*) common_ptr->endoCol);
	cudaFreeHost((void*) common_ptr->tEndoRowLoc);
	cudaFreeHost((void*) common_ptr->tEndoColLoc);

	cudaFree(common_ptr->d_endoRow);
	cudaFree(common_ptr->d_endoCol);
	cudaFree(common_ptr->d_tEndoRowLoc);
	cudaFree(common_ptr->d_tEndoColLoc);

	cudaFree(common_ptr->d_endoT);

	// epi points
	cudaFreeHost((void*) common_ptr->epiRow);
	cudaFreeHost((void*) common_ptr->epiCol);
	cudaFreeHost((void*) common_ptr->tEpiRowLoc);
	cudaFreeHost((void*) common_ptr->tEpiColLoc);

	cudaFree(common_ptr->d_epiRow);
	cudaFree(common_ptr->d_epiCol);
	cudaFree(common_ptr->d_tEpiRowLoc);
	cudaFree(common_ptr->d_tEpiColLoc);

	cudaFree(common_ptr->d_epiT);

    cudaFreeHost((void*) common_ptr);
    cudaFreeHost((void*) common_change_ptr);

	//====================================================================================================
	//	POINTERS
	//====================================================================================================

	for(i=0; i<common_ptr->allPoints; i++){
		cudaFree(unique_ptr[i].d_in2);

		cudaFree(unique_ptr[i].d_conv);
		cudaFree(unique_ptr[i].d_in2_pad_cumv);
		cudaFree(unique_ptr[i].d_in2_pad_cumv_sel);
		cudaFree(unique_ptr[i].d_in2_sub_cumh);
		cudaFree(unique_ptr[i].d_in2_sub_cumh_sel);
		cudaFree(unique_ptr[i].d_in2_sub2);
		cudaFree(unique_ptr[i].d_in2_sqr);
		cudaFree(unique_ptr[i].d_in2_sqr_sub2);
		cudaFree(unique_ptr[i].d_in_sqr);

		cudaFree(unique_ptr[i].d_tMask);
		cudaFree(unique_ptr[i].d_mask_conv);
	}

    cudaFreeHost((void*) unique_ptr);
}

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	MAIN FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
