#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "oclengine.h"
#include "winglue.h"
#include <cassert>

OCLEngine::OCLEngine(int platform_id, int device_id, const char* program, uint32_t ncols,
	uint32_t nrows, uint32_t invsize, bool is_unlim_round, int32_t addr_mode, const char* pkey_base,
	const char* filename, bool& should_exit) :
	_is_unlim_round(is_unlim_round), _addr_mode(addr_mode), _pkey_base(pkey_base)
{

	READY = false;
	struct timeval before {}, after{};
	uint8_t buf[20];
	FILE* wfd;
	uint64_t N = 0;

	gettimeofday(&before, nullptr);
	wfd = fopen(filename, "rb");
	if (!wfd) {
		printf("%s can not open\n", filename);
		exit2("bloom init", 1);
	}

	_fseeki64(wfd, 0, SEEK_END);
	N = _ftelli64(wfd);
	N = N / 20;
	BLOOM_N = 2 * N;
	rewind(wfd);

	auto* heap = (uint8_t*)malloc(N * 20);
	memset(heap, 0, N * 20);

	_bloom = new Bloom(BLOOM_N, 0.00001);

	uint64_t percent = (N - 1) / 100;
	uint64_t i = 0;
	while (i < N && !should_exit) {
		memset(buf, 0, 20);
		memset(heap + (i * 20), 0, 20);
		if (fread(buf, 1, 20, wfd) == 20) {
			_bloom->add(buf, 20);
			memcpy(heap + (i * 20), buf, 20);
			if (i % percent == 0) {
				printf("\rLoading addresses: %llu %%", (i / percent));
				fflush(stdout);
			}
		}
		i++;
	}
	if (should_exit)
		exit2("", 0);

	printf("\n");
	fclose(wfd);
	BLOOM_N = _bloom->get_bytes();
	DATA = heap;
	DATA_SIZE = N * 20;

	gettimeofday(&after, nullptr);
	printf("Loaded addresses : %llu in %01.6f sec\n", i, (double)(time_diff(before, after) / 1000000));
	printf("\n");
	_bloom->print(); 
	printf("\n");

	/////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////

	/* get available platforms */
	if ((_platform_id = ocl_platform_get(platform_id)) == nullptr) {
		exit2("ocl_platform_get", 1);
	}

	/* get available devices */
	if ((_device_id = ocl_device_get(_platform_id, device_id)) == nullptr) {
		exit2("ocl_device_get", 1);
	}

	cl_int ret;

	/* create context */
	_context = clCreateContext(nullptr, 1, &(_device_id), nullptr, nullptr, &ret);
	if (!_context) {
		ocl_error(ret, "clCreateContext");
		exit2("clCreateContext", 1);
	}

	/* create a command */
	_command = clCreateCommandQueue(_context, _device_id, 0, &ret);
	if (!_command) {
		ocl_error(ret, "clCreateCommandQueue");
		exit2("clCreateCommandQueue", 1);
	}


	/* get compiler options */
	char optbuf[256];
	_quirks = ocl_get_quirks(_device_id, optbuf);

	/*Loading and compiling a CL program*/
	if (!ocl_load_program(program, optbuf)) {
		exit2("ocl_load_program", 1);
	}


	/*Calculating Matrix Settings*/

	/*Number of simultaneously executed threads per GPU */
	size_t nthreads = ocl_device_getsizet(_device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE);
	size_t full_threads = ocl_device_getsizet(_device_id, CL_DEVICE_MAX_COMPUTE_UNITS);
	full_threads *= nthreads;

	cl_ulong memsize = ocl_device_getulong(_device_id, CL_DEVICE_GLOBAL_MEM_SIZE);
	cl_ulong allocsize = ocl_device_getulong(_device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
	memsize /= 2;

	if (!ncols || !nrows) {

		ncols = full_threads;
		nrows = 2;
		while ((ncols > nrows) && !(ncols & 1)) {
			ncols /= 2;
			nrows *= 2;
		}

		int worksize = 2048; //defult is 2048
		int wsmult = 1;
		while ((!worksize || ((wsmult * 2) <= worksize)) &&
			((ncols * nrows * 2 * 128) < memsize) &&
			((ncols * nrows * 2 * 64) < allocsize)) {
			if (ncols > nrows)
				nrows *= 2;
			else
				ncols *= 2;
			wsmult *= 2;
		}

	}

	uint32_t round = nrows * ncols;

	if (!invsize) {
		invsize = 2;
		while (!(round % (invsize << 1)) && ((round / invsize) > full_threads))
			invsize <<= 1;
	}

	if ((round % invsize) || !is_pow2(invsize) || (invsize < 2)) {
		fprintf(stderr, "Grid size: %dx%d\n", ncols, nrows);
		fprintf(stderr, "Modular inverse thread size: %d\n", invsize);
		if (round % invsize)
			fprintf(stderr, "Modular inverse work size must evenly divide points\n");
		else
			fprintf(stderr, "Modular inverse work per task (%d) must be a power of 2\n", invsize);
		exit2("Grid size settings", 1);
	}

	_ncols = ncols;
	_nrows = nrows;
	_round = round;
	_invsize = invsize;

	printf("\n\n");
	printf("MATRIX:\n");
	printf("\tGrid size  : %dx%d\n", ncols, nrows);
	printf("\tTotal      : %d\n", round);
	printf("\tMod inverse: %d threads [%d ops/thread]\n", round / invsize, invsize);

	ocl_kernel_init();
	ocl_print_info();

	READY = true;
}

OCLEngine::~OCLEngine()
{

	int i, arg;
	for (arg = 0; arg < MAX_ARG; arg++) {
		if (_arguments[arg]) {
			clReleaseMemObject(_arguments[arg]);
			_arguments[arg] = nullptr;
			_argument_size[arg] = 0;
		}
	}
	for (i = 0; i < MAX_KERNEL; i++) {
		if (_kernel[i]) {
			clReleaseKernel(_kernel[i]);
			_kernel[i] = nullptr;
		}
	}

	if (_program) {
		clReleaseProgram(_program);
	}
	if (_command) {
		clReleaseCommandQueue(_command);
	}
	if (_context) {
		clReleaseContext(_context);
	}

	if (DATA)
		free(DATA);
	delete _bloom;
}

void OCLEngine::exit2(const char* err, int ret)
{
	fprintf(stderr, "\nERROR: (%d) : %s\n", ret, err);
	exit(ret);
}

bool OCLEngine::is_ready() const
{
	return READY;
}

void OCLEngine::loop(bool& should_exit)
{
	int i, n;

	BIGNUM* bn_tmp = BN_new();
	BN_CTX* bn_ctx = BN_CTX_new();

	EC_KEY* pkey = EC_KEY_new_by_curve_name(NID_secp256k1);
	EC_KEY_precompute_mult(pkey, bn_ctx);

	BIGNUM* N = BN_new();
	BN_hex2bn(&N, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

	const BIGNUM* bn_key = NULL;

	const EC_GROUP* pgroup = EC_KEY_get0_group(pkey);
	const EC_POINT* pgen = EC_GROUP_get0_generator(pgroup);

	EC_POINT** pprows = NULL;
	EC_POINT** ppcols = NULL;
	EC_POINT* pbatchinc = NULL;
	EC_POINT* poffset = NULL;

	//Allocating memory for matrix base points
	ppcols = (EC_POINT**)malloc(_ncols * sizeof(EC_POINT*));
	pprows = (EC_POINT**)malloc(_nrows * sizeof(EC_POINT*));
	for (i = 0; i < (int)_ncols; i++) {
		ppcols[i] = EC_POINT_new(pgroup);
	}
	for (i = 0; i < (int)_nrows; i++) {
		pprows[i] = EC_POINT_new(pgroup);
	}

	pbatchinc = EC_POINT_new(pgroup);
	poffset = EC_POINT_new(pgroup);

	BN_set_word(bn_tmp, _ncols);
	EC_POINT_mul(pgroup, pbatchinc, bn_tmp, NULL, NULL, bn_ctx);
	EC_POINT_make_affine(pgroup, pbatchinc, bn_ctx);

	//The point to shift the initial increments by the total number of elements in the matrix
	BN_set_word(bn_tmp, _round);
	EC_POINT_mul(pgroup, poffset, bn_tmp, NULL, NULL, bn_ctx);
	EC_POINT_make_affine(pgroup, poffset, bn_ctx);


	uint64_t       total = 0;
	uint32_t       iterations = 0;        //Number of private key change iterations
	uint32_t       rounds = 0;        //Number of rounds of work with GPU
	time_t         now = 0;
	uint32_t       found_deltau = 0;
	uint32_t       found_deltac = 0;
	//uint32_t       found_pos    = 0;
	uint8_t* points_in = NULL;
	uint8_t* strides_in = NULL;
	uint32_t* uint32_ptr = NULL;
	uint8_t* uint8_ptr = NULL;
	uint8_t* found_hashu = NULL;
	uint8_t* found_hashc = NULL;
	KeyInfo* info = NULL;
	FILE* ffd = NULL;
	uint8_t        pkey_bin[32];
	uint8_t        pkey_s[65];
	char           buffer[4096];
	char           time_buf[128];
	uint8_t        hash_buf[128];
	char           tmp[1024];

	HashRate round_hr;
	HashRate total_hr;

	uint32_t round_max = (_is_unlim_round == false ? (uint32_t)(0xFFFFFFFF / _round) + 1 : 0);

	gettimeofday(&(total_hr.time_start), NULL);

	while (!should_exit) {
		/******************************************************************/

		iterations++;

		//Setting the result buffer to its default position
		uint32_ptr = (uint32_t*)ocl_map_arg_buffer(0, 1);
		if (!uint32_ptr) {
			fprintf(stderr, "ERROR: Could not map result buffer\n");
			return;
		}
		uint32_ptr[0] = 0xffffffff;
		//uint32_ptr[6] = 0xffffffff;
		ocl_unmap_arg_buffer(0, uint32_ptr);


		//Generating a random private key
		EC_KEY_generate_key(pkey);


		//If the starting private key is set, set it
		if (iterations == 1 && strlen(_pkey_base) != 0) {
			BN_hex2bn(&bn_tmp, _pkey_base);
			Utils::set_pkey(bn_tmp, pkey);
		}

		//Displaying the key on the screen
		bn_key = EC_KEY_get0_private_key(pkey);
		n = BN_num_bytes(bn_key);
		if (n < 32) {
			memset(pkey_bin, 0, 32 - n);
		}
		BN_bn2bin(bn_key, &pkey_bin[32 - n]);
		Utils::bin2hex(pkey_s, pkey_bin, 32);

		now = time(NULL);
		strftime(buffer, 1023, "%Y-%m-%d %H:%M:%S", localtime(&now));
		printf("\nIteration %u at [%s] from: %s\n", iterations, buffer, pkey_s);


		//Preparing initial values for the matrix
		EC_POINT_copy(ppcols[0], EC_KEY_get0_public_key(pkey));

		//Preparing initial values for the matrix
		for (i = 1; i < (int)_ncols; i++) {
			EC_POINT_add(pgroup, ppcols[i], ppcols[i - 1], pgen, bn_ctx);
		}
		EC_POINTs_make_affine(pgroup, _ncols, ppcols, bn_ctx);

		//Fill in the obtained base points the variables of the OpenCL function
		points_in = (uint8_t*)ocl_map_arg_buffer(3, 1);
		if (!points_in) {
			fprintf(stderr, "ERROR: Could not map column buffer\n"); return;
		}
		for (i = 0; i < (int)_ncols; i++) {
			ocl_put_point_tpa(points_in, i, ppcols[i]);
		}
		ocl_unmap_arg_buffer(3, points_in);

		//Calculating incremental base points
		EC_POINT_copy(pprows[0], pgen);
		for (i = 1; i < (int)_nrows; i++) {
			EC_POINT_add(pgroup, pprows[i], pprows[i - 1], pbatchinc, bn_ctx);
		}
		EC_POINTs_make_affine(pgroup, _nrows, pprows, bn_ctx);

		rounds = 1;

		while ((rounds < round_max || round_max == 0) && !should_exit) {

			gettimeofday(&(round_hr.time_start), NULL);

			bn_key = EC_KEY_get0_private_key(pkey);
			n = BN_num_bytes(bn_key);
			if (n < 32) {
				memset(pkey_bin, 0, 32 - n);
			}
			BN_bn2bin(bn_key, &pkey_bin[32 - n]);
			Utils::bin2hex(pkey_s, pkey_bin, 32);
			//printf("\nround %u from: %s\n",rounds, pkey_s);

			if (rounds > 1) {
				//Shift the increment by poffset points forward
				for (i = 0; i < (int)_nrows; i++) {
					EC_POINT_add(pgroup, pprows[i], pprows[i], poffset, bn_ctx);
				}
				EC_POINTs_make_affine(pgroup, _nrows, pprows, bn_ctx);
			}

			//Copying Incremental Base Points to a Device
			strides_in = (uint8_t*)ocl_map_arg_buffer(4, 1);
			if (!strides_in) {
				fprintf(stderr, "ERROR: Could not map row buffer\n"); return;
			}
			memset(strides_in, 0, 64 * _nrows);
			for (i = 0; i < (int)_nrows; i++) {
				ocl_put_point(strides_in + (64 * i), pprows[i]);
			}
			ocl_unmap_arg_buffer(4, strides_in);

			if (ocl_kernel_start()) {
				//Getting the value of the attribute of finding a match
				uint8_ptr = (uint8_t*)ocl_map_arg_buffer(0, 2);
				if (!uint8_ptr) {
					fprintf(stderr, "ERROR: Could not map result buffer");
					return;
				}

				if (_addr_mode == 0) {

					found_deltau = ((uint32_t*)uint8_ptr)[0];

					if (found_deltau != 0xffffffff) {
						found_hashu = &uint8_ptr[0 + 4];
						if (check_hash_binary(found_hashu) > 0) {
							report(bn_tmp, bn_key, info, found_deltau, found_hashu, hash_buf,
								&now, time_buf, buffer, tmp, pkey_s, ffd, UNCOMPRESSED);
						}
						memset(uint8_ptr, 0, ARG_FOUND_SIZE / 2);
						memset(uint8_ptr, 0xFF, 4);
					}

				}
				else if (_addr_mode == 1) {

					found_deltac = ((uint32_t*)uint8_ptr)[6];

					if (found_deltac != 0xffffffff) {
						found_hashc = &uint8_ptr[24 + 4];
						if (check_hash_binary(found_hashc) > 0) {
							report(bn_tmp, bn_key, info, found_deltac, found_hashc, hash_buf,
								&now, time_buf, buffer, tmp, pkey_s, ffd, COMPRESSED);
						}
						memset(uint8_ptr + 24, 0, ARG_FOUND_SIZE / 2);
						memset(uint8_ptr + 24, 0xFF, 4);
					}
				}
				else {
					found_deltau = ((uint32_t*)uint8_ptr)[0];
					found_deltac = ((uint32_t*)uint8_ptr)[6];

					if (found_deltau != 0xffffffff) {
						found_hashu = &uint8_ptr[0 + 4];
						if (check_hash_binary(found_hashu) > 0) {
							report(bn_tmp, bn_key, info, found_deltau, found_hashu, hash_buf,
								&now, time_buf, buffer, tmp, pkey_s, ffd, UNCOMPRESSED);
						}
						memset(uint8_ptr, 0, ARG_FOUND_SIZE / 2);
						memset(uint8_ptr, 0xFF, 4);
					}

					if (found_deltac != 0xffffffff) {
						found_hashc = &uint8_ptr[24 + 4];
						if (check_hash_binary(found_hashc) > 0) {
							report(bn_tmp, bn_key, info, found_deltac, found_hashc, hash_buf,
								&now, time_buf, buffer, tmp, pkey_s, ffd, COMPRESSED);
						}
						memset(uint8_ptr + 24, 0, ARG_FOUND_SIZE / 2);
						memset(uint8_ptr + 24, 0xFF, 4);
					}
				}
				ocl_unmap_arg_buffer(0, uint8_ptr);

				//private key increment
				BN_copy(bn_tmp, bn_key);
				BN_add_word(bn_tmp, _round);
				Utils::set_pkey(bn_tmp, pkey);
			}
			else {
				return;
			}

			total += _round;

			Utils::hashrate_update(&round_hr, _round);
			Utils::hashrate_update(&total_hr, total);

			printf("\r[%s] [round %u: %01.2fs (%01.2f %s)] [total %s (%01.2f %s)]   ",
				pkey_s, rounds, round_hr.runtime, round_hr.hashrate, round_hr.unit, formatThousands(total).c_str(), total_hr.hashrate, total_hr.unit);
			fflush(stdout);

			rounds++;
		}
	}
	return;
}

std::string OCLEngine::formatThousands(uint64_t x)
{
	char buf[32] = "";

	sprintf(buf, "%lld", x);

	std::string s(buf);

	int len = (int)s.length();

	int numCommas = (len - 1) / 3;

	if (numCommas == 0) {
		return s;
	}

	std::string result = "";

	int count = ((len % 3) == 0) ? 0 : (3 - (len % 3));

	for (int i = 0; i < len; i++) {
		result += s[i];

		if (count++ == 2 && i < len - 1) {
			result += ",";
			count = 0;
		}
	}

	return result;
}

void OCLEngine::report(BIGNUM* bn_tmp, const BIGNUM* bn_key, KeyInfo* info, uint32_t found_delta, const uint8_t* found_hash,
	uint8_t* hash_buf, time_t* now, char* time_buf, char* buffer, char* tmp, uint8_t* pkey_s, FILE* ffd, PubType pubtype)
{
	BN_copy(bn_tmp, bn_key);
	BN_add_word(bn_tmp, found_delta + 1);

	info = Utils::get_key_info(bn_tmp, pubtype);
	Utils::bin2hex(hash_buf, found_hash, 20);
	*now = time(NULL);
	strftime(time_buf, 127, "%Y-%m-%d %H:%M:%S", localtime(now));

	int n = sprintf(buffer,
		"\n++++++++++++++++++++++++++++++++++++++++++++++++++\n"\
		"TIME: %s\n"\
		"PRIV: %s\n"\
		"PUBK: %s\n"\
		"HASH: %s\n"\
		"ADDR: %s\n"\
		"SALT: %s\n"\
		"OFST: %i\n"\
		"GPUH: %s\n"\
		"++++++++++++++++++++++++++++++++++++++++++++++++++\n",
		time_buf,
		info->private_hex,
		pubtype == COMPRESSED ? info->publicc_hex : info->publicu_hex,
		info->public_ripemd160_hex,
		info->address_hex,
		pkey_s,
		found_delta,
		hash_buf
	);
	printf("\n%s\n", buffer);
	sprintf(tmp, "./%s.%u.txt", info->public_ripemd160_hex, (uint32_t)*now);
	ffd = fopen(tmp, "w");
	if (ffd) {
		fwrite(buffer, n, 1, ffd);
		fclose(ffd);
	}
	ffd = fopen("./found.txt", "a");
	if (ffd) {
		fwrite(buffer, n, 1, ffd);
		fclose(ffd);
	}
	free(info);
}

/***********************************************************************
 * OpenCL debugging and support
 ***********************************************************************/

const char* OCLEngine::ocl_strerror(cl_int ret)
{
#define OCL_STATUS(st) case st: return #st;
	switch (ret) {
		OCL_STATUS(CL_SUCCESS);
		OCL_STATUS(CL_DEVICE_NOT_FOUND);
		OCL_STATUS(CL_DEVICE_NOT_AVAILABLE);
		OCL_STATUS(CL_COMPILER_NOT_AVAILABLE);
		OCL_STATUS(CL_MEM_OBJECT_ALLOCATION_FAILURE);
		OCL_STATUS(CL_OUT_OF_RESOURCES);
		OCL_STATUS(CL_OUT_OF_HOST_MEMORY);
		OCL_STATUS(CL_PROFILING_INFO_NOT_AVAILABLE);
		OCL_STATUS(CL_MEM_COPY_OVERLAP);
		OCL_STATUS(CL_IMAGE_FORMAT_MISMATCH);
		OCL_STATUS(CL_IMAGE_FORMAT_NOT_SUPPORTED);
		OCL_STATUS(CL_BUILD_PROGRAM_FAILURE);
		OCL_STATUS(CL_MAP_FAILURE);
#if defined(CL_MISALIGNED_SUB_BUFFER_OFFSET)
		OCL_STATUS(CL_MISALIGNED_SUB_BUFFER_OFFSET);
#endif /* defined(CL_MISALIGNED_SUB_BUFFER_OFFSET) */
#if defined(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
		OCL_STATUS(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
#endif /* defined(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST) */
		OCL_STATUS(CL_INVALID_VALUE);
		OCL_STATUS(CL_INVALID_DEVICE_TYPE);
		OCL_STATUS(CL_INVALID_PLATFORM);
		OCL_STATUS(CL_INVALID_DEVICE);
		OCL_STATUS(CL_INVALID_CONTEXT);
		OCL_STATUS(CL_INVALID_QUEUE_PROPERTIES);
		OCL_STATUS(CL_INVALID_COMMAND_QUEUE);
		OCL_STATUS(CL_INVALID_HOST_PTR);
		OCL_STATUS(CL_INVALID_MEM_OBJECT);
		OCL_STATUS(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
		OCL_STATUS(CL_INVALID_IMAGE_SIZE);
		OCL_STATUS(CL_INVALID_SAMPLER);
		OCL_STATUS(CL_INVALID_BINARY);
		OCL_STATUS(CL_INVALID_BUILD_OPTIONS);
		OCL_STATUS(CL_INVALID_PROGRAM);
		OCL_STATUS(CL_INVALID_PROGRAM_EXECUTABLE);
		OCL_STATUS(CL_INVALID_KERNEL_NAME);
		OCL_STATUS(CL_INVALID_KERNEL_DEFINITION);
		OCL_STATUS(CL_INVALID_KERNEL);
		OCL_STATUS(CL_INVALID_ARG_INDEX);
		OCL_STATUS(CL_INVALID_ARG_VALUE);
		OCL_STATUS(CL_INVALID_ARG_SIZE);
		OCL_STATUS(CL_INVALID_KERNEL_ARGS);
		OCL_STATUS(CL_INVALID_WORK_DIMENSION);
		OCL_STATUS(CL_INVALID_WORK_GROUP_SIZE);
		OCL_STATUS(CL_INVALID_WORK_ITEM_SIZE);
		OCL_STATUS(CL_INVALID_GLOBAL_OFFSET);
		OCL_STATUS(CL_INVALID_EVENT_WAIT_LIST);
		OCL_STATUS(CL_INVALID_EVENT);
		OCL_STATUS(CL_INVALID_OPERATION);
		OCL_STATUS(CL_INVALID_GL_OBJECT);
		OCL_STATUS(CL_INVALID_BUFFER_SIZE);
		OCL_STATUS(CL_INVALID_MIP_LEVEL);
		OCL_STATUS(CL_INVALID_GLOBAL_WORK_SIZE);
#if defined(CL_INVALID_PROPERTY)
		OCL_STATUS(CL_INVALID_PROPERTY);
#endif /* defined(CL_INVALID_PROPERTY) */
#undef OCL_STATUS
	default: {
		static char tmp[64];
		snprintf(tmp, sizeof(tmp), "Unknown code %d", ret);
		return tmp;
	}
	}
}

void OCLEngine::ocl_error(int code, const char* desc)
{
	const char* err = ocl_strerror(code);
	if (desc) {
		fprintf(stderr, "%s: %s\n", desc, err);
	}
	else {
		fprintf(stderr, "%s\n", err);
	}
}


/*Displaying information about the current device and platform*/
void OCLEngine::ocl_print_info()
{
	cl_device_id did = _device_id;

	printf("\nDEVICE INFO:\n");
	printf("\tDevice              : %s\n", ocl_device_getstr(did, CL_DEVICE_NAME));
	printf("\tVendor              : %s (%04x)\n", ocl_device_getstr(did, CL_DEVICE_VENDOR), ocl_device_getuint(did, CL_DEVICE_VENDOR_ID));
	printf("\tDriver              : %s\n", ocl_device_getstr(did, CL_DRIVER_VERSION));
	printf("\tProfile             : %s\n", ocl_device_getstr(did, CL_DEVICE_PROFILE));
	printf("\tVersion             : %s\n", ocl_device_getstr(did, CL_DEVICE_VERSION));
	printf("\tMax compute units   : %zd\n", ocl_device_getsizet(did, CL_DEVICE_MAX_COMPUTE_UNITS));
	printf("\tMax workgroup size  : %zd\n", ocl_device_getsizet(did, CL_DEVICE_MAX_WORK_GROUP_SIZE));
	printf("\tGlobal memory       : %llu\n", ocl_device_getulong(did, CL_DEVICE_GLOBAL_MEM_SIZE));
	printf("\tMax allocation      : %llu\n\n", ocl_device_getulong(did, CL_DEVICE_MAX_MEM_ALLOC_SIZE));
}


/***********************************************************************
 * PLATFORM
 ***********************************************************************/

 /*Getting the platform*/
cl_platform_id OCLEngine::ocl_platform_get(int num)
{

	int np;
	cl_platform_id id, * ids;

	np = ocl_platform_list(&ids);
	if (np < 0)
		return nullptr;
	if (!np) {
		fprintf(stderr, "No OpenCL platforms available\n");
		return nullptr;
	}
	if (num < 0) {
		if (np == 1)
			num = 0;
		else
			num = np;
	}
	if (num < np) {
		id = ids[num];
		free(ids);
		return id;
	}
	free(ids);
	return nullptr;
}


/*Getting a list of available platforms*/
int OCLEngine::ocl_platform_list(cl_platform_id** list_out)
{
	cl_uint np;
	cl_int res;
	cl_platform_id* ids;
	res = clGetPlatformIDs(0, nullptr, &np);
	if (res != CL_SUCCESS) {
		ocl_error(res, "clGetPlatformIDs(0)");
		*list_out = nullptr;
		return -1;
	}
	if (np) {
		ids = (cl_platform_id*)malloc(np * sizeof(cl_platform_id));
		if (ids == nullptr) {
			fprintf(stderr,
				"Could not allocate platform ID list\n");
			*list_out = nullptr;
			return -1;
		}
		res = clGetPlatformIDs(np, ids, nullptr);
		if (res != CL_SUCCESS) {
			ocl_error(res, "clGetPlatformIDs(n)");
			free(ids);
			*list_out = nullptr;
			return -1;
		}
		*list_out = ids;
	}
	return np;
}

/*Displaying information about available platforms*/
void OCLEngine::ocl_platforms_info(cl_platform_id* ids, int np, int base)
{
	int i;
	char nbuf[128];
	char vbuf[128];
	size_t len;
	cl_int res;

	for (i = 0; i < np; i++) {
		res = clGetPlatformInfo(ids[i], CL_PLATFORM_NAME, sizeof(nbuf), nbuf, &len);
		if (res != CL_SUCCESS) {
			ocl_error(res, "clGetPlatformInfo(NAME)");
			continue;
		}
		if (len >= sizeof(nbuf))
			len = sizeof(nbuf) - 1;
		nbuf[len] = '\0';
		res = clGetPlatformInfo(ids[i], CL_PLATFORM_VENDOR, sizeof(vbuf), vbuf, &len);
		if (res != CL_SUCCESS) {
			ocl_error(res, "clGetPlatformInfo(VENDOR)");
			continue;
		}
		if (len >= sizeof(vbuf))
			len = sizeof(vbuf) - 1;
		vbuf[len] = '\0';
		fprintf(stderr, "%d: [%s] %s\n", i + base, vbuf, nbuf);
	}
}


const char* OCLEngine::ocl_platform_getstr(cl_platform_id pid, cl_platform_info param)
{
	static char platform_str[1024];
	cl_int ret;
	size_t size_ret;
	ret = clGetPlatformInfo(pid, param,
		sizeof(platform_str), platform_str,
		&size_ret);
	if (ret != CL_SUCCESS) {
		snprintf(platform_str, sizeof(platform_str),
			"clGetPlatformInfo(%d): %s",
			param, ocl_strerror(ret));
	}
	return platform_str;
}


/***********************************************************************
 * DEVICE
 ***********************************************************************/

 /*Getting the specified device on the specified platform*/
cl_device_id OCLEngine::ocl_device_manual(int platformidx, int deviceidx)
{
	cl_platform_id pid;
	cl_device_id did = nullptr;

	pid = ocl_platform_get(platformidx);
	if (pid) {
		did = ocl_device_get(pid, deviceidx);
		if (did)
			return did;
	}
	return nullptr;
}

/*Receiving the device*/
cl_device_id OCLEngine::ocl_device_get(cl_platform_id pid, int num)
{
	int nd;
	cl_device_id id, * ids;

	nd = ocl_devices_list(pid, &ids);
	if (nd < 0)
		return nullptr;
	if (!nd) {
		fprintf(stderr, "No OpenCL devices found\n");
		return nullptr;
	}
	if (num < 0) {
		if (nd == 1)
			num = 0;
		else
			num = nd;
	}
	if (num < nd) {
		id = ids[num];
		free(ids);
		return id;
	}
	free(ids);
	return nullptr;
}


/*Platform device list*/
int OCLEngine::ocl_devices_list(cl_platform_id pid, cl_device_id** list_out)
{
	cl_uint nd;
	cl_int res;
	cl_device_id* ids;
	res = clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
	if (res != CL_SUCCESS) {
		ocl_error(res, "clGetDeviceIDs(0)");
		*list_out = nullptr;
		return -1;
	}
	if (nd) {
		ids = (cl_device_id*)malloc(nd * sizeof(cl_device_id));
		if (ids == nullptr) {
			fprintf(stderr, "Could not allocate device ID list\n");
			*list_out = nullptr;
			return -1;
		}
		res = clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, nd, ids, nullptr);
		if (res != CL_SUCCESS) {
			ocl_error(res, "clGetDeviceIDs(n)");
			free(ids);
			*list_out = nullptr;
			return -1;
		}
		*list_out = ids;
	}
	return nd;
}


/*Platform device information*/
void OCLEngine::ocl_devices_info(cl_platform_id pid, cl_device_id* ids, int nd, int base)
{
	int i;
	char nbuf[128];
	char vbuf[128];
	size_t len;
	cl_int res;

	(void)pid;

	for (i = 0; i < nd; i++) {
		res = clGetDeviceInfo(ids[i], CL_DEVICE_NAME, sizeof(nbuf), nbuf, &len);
		if (res != CL_SUCCESS)
			continue;
		if (len >= sizeof(nbuf))
			len = sizeof(nbuf) - 1;
		nbuf[len] = '\0';
		res = clGetDeviceInfo(ids[i], CL_DEVICE_VENDOR, sizeof(vbuf), vbuf, &len);
		if (res != CL_SUCCESS)
			continue;
		if (len >= sizeof(vbuf))
			len = sizeof(vbuf) - 1;
		vbuf[len] = '\0';
		fprintf(stderr, "  %d: [%s] %s\n", i + base, vbuf, nbuf);
	}
}

/*Returns the platform of the given device*/
cl_platform_id OCLEngine::ocl_device_getplatform(cl_device_id did)
{
	cl_int ret;
	cl_platform_id val;
	size_t size_ret;
	ret = clGetDeviceInfo(did, CL_DEVICE_PLATFORM,
		sizeof(cl_platform_id), &val, &size_ret);
	if (ret != CL_SUCCESS) {
		fprintf(stderr, "clGetDeviceInfo(CL_DEVICE_PLATFORM): %s",
			ocl_strerror(ret));
	}
	return val;
}

/*Returns the type of device*/
cl_device_type OCLEngine::ocl_device_gettype(cl_device_id did)
{
	cl_int ret;
	cl_device_type val;
	size_t size_ret;
	ret = clGetDeviceInfo(did, CL_DEVICE_TYPE,
		sizeof(val), &val, &size_ret);
	if (ret != CL_SUCCESS) {
		fprintf(stderr, "clGetDeviceInfo(CL_DEVICE_TYPE): %s",
			ocl_strerror(ret));
	}
	return val;
}

/*Returns the text parameter of the device*/
const char* OCLEngine::ocl_device_getstr(cl_device_id did, cl_device_info param)
{
	static char device_str[1024];
	cl_int ret;
	size_t size_ret;
	ret = clGetDeviceInfo(did, param,
		sizeof(device_str), device_str,
		&size_ret);
	if (ret != CL_SUCCESS) {
		snprintf(device_str, sizeof(device_str),
			"clGetDeviceInfo(%d): %s",
			param, ocl_strerror(ret));
	}
	return device_str;
}

/*Returns the size_t device parameter*/
size_t OCLEngine::ocl_device_getsizet(cl_device_id did, cl_device_info param)
{
	cl_int ret;
	size_t val = 0;
	size_t size_ret;
	ret = clGetDeviceInfo(did, param, sizeof(val), &val, &size_ret);
	if (ret != CL_SUCCESS) {
		fprintf(stderr,
			"clGetDeviceInfo(%d): %s", param, ocl_strerror(ret));
	}
	return val;
}

/*Returns the cl_ulong device parameter*/
cl_ulong OCLEngine::ocl_device_getulong(cl_device_id did, cl_device_info param)
{
	cl_int ret;
	cl_ulong val;
	size_t size_ret;
	ret = clGetDeviceInfo(did, param, sizeof(val), &val, &size_ret);
	if (ret != CL_SUCCESS) {
		fprintf(stderr,
			"clGetDeviceInfo(%d): %s", param, ocl_strerror(ret));
	}
	return val;
}

/*Returns the cl_uint device parameter*/
cl_uint OCLEngine::ocl_device_getuint(cl_device_id did, cl_device_info param)
{
	cl_int ret;
	cl_uint val;
	size_t size_ret;
	ret = clGetDeviceInfo(did, param, sizeof(val), &val, &size_ret);
	if (ret != CL_SUCCESS) {
		fprintf(stderr,
			"clGetDeviceInfo(%d): %s", param, ocl_strerror(ret));
	}
	return val;
}


enum {
	VG_OCL_DEEP_PREPROC_UNROLL = (1 << 0),
	VG_OCL_PRAGMA_UNROLL = (1 << 1),
	VG_OCL_EXPENSIVE_BRANCHES = (1 << 2),
	VG_OCL_DEEP_VLIW = (1 << 3),
	VG_OCL_AMD_BFI_INT = (1 << 4),
	VG_OCL_NV_VERBOSE = (1 << 5),
	VG_OCL_BROKEN = (1 << 6),
	VG_OCL_NO_BINARIES = (1 << 7),

	VG_OCL_OPTIMIZATIONS = (VG_OCL_DEEP_PREPROC_UNROLL |
	VG_OCL_PRAGMA_UNROLL |
		VG_OCL_EXPENSIVE_BRANCHES |
		VG_OCL_DEEP_VLIW |
		VG_OCL_AMD_BFI_INT),

};


/***********************************************************************
 * PROGRAM
 ***********************************************************************/

 /*Computation based on device options for compilation*/
unsigned int OCLEngine::ocl_get_quirks(cl_device_id did, char* optbuf)
{
	uint32_t vend;
	const char* dvn;
	unsigned int quirks = 0;

	quirks |= VG_OCL_DEEP_PREPROC_UNROLL;

	vend = ocl_device_getuint(did, CL_DEVICE_VENDOR_ID);
	switch (vend) {
	case 0x10de: /* NVIDIA */
		/*
		 * NVIDIA's compiler seems to take a really really long
		 * time when using preprocessor unrolling, but works
		 * well with pragma unroll.
		 */
		quirks &= ~VG_OCL_DEEP_PREPROC_UNROLL;
		quirks |= VG_OCL_PRAGMA_UNROLL;
		quirks |= VG_OCL_NV_VERBOSE;
		break;
	case 0x1002: /* AMD/ATI */
		/*
		 * AMD's compiler works best with preprocesor unrolling.
		 * Pragma unroll is unreliable with AMD's compiler and
		 * seems to crash based on whether the gods were smiling
		 * when Catalyst was last installed/upgraded.
		 */
		if (ocl_device_gettype(did) & CL_DEVICE_TYPE_GPU) {
			quirks |= VG_OCL_EXPENSIVE_BRANCHES;
			quirks |= VG_OCL_DEEP_VLIW;
			dvn = ocl_device_getstr(did, CL_DEVICE_EXTENSIONS);
			if (dvn && strstr(dvn, "cl_amd_media_ops"))
				quirks |= VG_OCL_AMD_BFI_INT;

			dvn = ocl_device_getstr(did, CL_DEVICE_NAME);
			if (!strcmp(dvn, "ATI RV710")) {
				quirks &= ~VG_OCL_OPTIMIZATIONS;
				quirks |= VG_OCL_NO_BINARIES;
			}
		}
		break;
	default:
		break;
	}

	if (optbuf) {
		ocl_get_quirks_str(quirks, optbuf);
	}

	return quirks;
}


/*Getting the compilation option string from the computed value ocl_get_quirks ()*/
void OCLEngine::ocl_get_quirks_str(unsigned int quirks, char* optbuf)
{
	int end = 0;
	if (quirks & VG_OCL_DEEP_PREPROC_UNROLL)
		end += sprintf(optbuf + end, "-DDEEP_PREPROC_UNROLL ");
	if (quirks & VG_OCL_PRAGMA_UNROLL)
		end += sprintf(optbuf + end, "-DPRAGMA_UNROLL ");
	if (quirks & VG_OCL_EXPENSIVE_BRANCHES)
		end += sprintf(optbuf + end, "-DVERY_EXPENSIVE_BRANCHES ");
	if (quirks & VG_OCL_DEEP_VLIW)
		end += sprintf(optbuf + end, "-DDEEP_VLIW ");
	if (quirks & VG_OCL_AMD_BFI_INT)
		end += sprintf(optbuf + end, "-DAMD_BFI_INT ");
	if (quirks & VG_OCL_NV_VERBOSE)
		end += sprintf(optbuf + end, "-cl-nv-verbose ");
	optbuf[end] = '\0';
}


int OCLEngine::ocl_load_program(const char* filename, const char* opts)
{
	FILE* kfp;
	char* buf, * tbuf;
	int len, fromsource = 0, patched = 0;
	size_t sz, szr;
	cl_program prog;
	cl_int ret, sts;
	uint32_t prog_hash = 0;
	char bin_name[64];
	uint8_t* ptr;

	sz = 128 * 1024;
	buf = (char*)malloc(sz);
	if (!buf) {
		fprintf(stderr, "Could not allocate program buffer\n");
		return 0;
	}

	fprintf(stderr, "Loading program file: %s\n", filename);
	kfp = fopen(filename, "r");
	if (!kfp) {
		fprintf(stderr, "Error loading kernel file '%s': %s\n",
			filename, strerror(errno));
		free(buf);
		return 0;
	}

	len = fread(buf, 1, sz, kfp);
	fclose(kfp);
	kfp = nullptr;

	if (!len) {
		fprintf(stderr, "Short read on CL kernel\n");
		free(buf);
		return 0;
	}

	prog_hash = ocl_hash_program(opts, buf, len);
	ptr = (uint8_t*)&prog_hash;
	sprintf(bin_name, "%02x%02x%02x%02x.oclbin", ptr[0], ptr[1], ptr[2], ptr[3]);

	if (_quirks & VG_OCL_NO_BINARIES) {
		//
	}
	else {
		kfp = fopen(bin_name, "rb");
	}

	//No binary, compile from source
	if (!kfp) {
		fromsource = 1;
		sz = len;
		prog = clCreateProgramWithSource(_context,
			1, (const char**)&buf, &sz,
			&ret);
	}
	else {
		szr = 0;
		while (!feof(kfp)) {
			len = fread(buf + szr, 1, sz - szr, kfp);
			if (!len) {
				fprintf(stderr,
					"Short read on CL kernel binary\n");
				fclose(kfp);
				free(buf);
				return 0;
			}
			szr += len;
			if (szr == sz) {
				tbuf = (char*)realloc(buf, sz * 2);
				if (!tbuf) {
					fprintf(stderr,
						"Could not expand CL kernel "
						"binary buffer\n");
					fclose(kfp);
					free(buf);
					return 0;
				}
				buf = tbuf;
				sz *= 2;
			}
		}
		fclose(kfp);
	rebuild:
		prog = clCreateProgramWithBinary(_context,
			1, &_device_id,
			&szr,
			(const unsigned char**)&buf,
			&sts,
			&ret);
	}
	free(buf);
	if (!prog) {
		ocl_error(ret, "clCreateProgramWithSource");
		return 0;
	}

	if (fromsource && !patched) {
		fprintf(stderr, "Compiling CL, can take minutes...");
		fflush(stderr);
	}

	ret = clBuildProgram(prog, 1, &_device_id, opts, nullptr, nullptr);
	if (ret != CL_SUCCESS) {
		if (fromsource && !patched)
			fprintf(stderr, "failure.\n");
		ocl_error(ret, "clBuildProgram");
		ocl_buildlog(prog);
		clReleaseProgram(prog);
		return 0;
	}

	if (fromsource && !(_quirks & VG_OCL_NO_BINARIES)) {
		ret = clGetProgramInfo(prog,
			CL_PROGRAM_BINARY_SIZES,
			sizeof(szr), &szr,
			&sz);
		if (ret != CL_SUCCESS) {
			ocl_error(ret,
				"WARNING: clGetProgramInfo(BINARY_SIZES)");
			goto out;
		}
		if (sz == 0) {
			fprintf(stderr,
				"WARNING: zero-length CL kernel binary\n");
			goto out;
		}

		buf = (char*)malloc(szr);
		if (!buf) {
			fprintf(stderr,
				"WARNING: Could not allocate %zd bytes "
				"for CL binary\n",
				szr);
			goto out;
		}

		ret = clGetProgramInfo(prog,
			CL_PROGRAM_BINARIES,
			sizeof(buf), &buf,
			&sz);
		if (ret != CL_SUCCESS) {
			ocl_error(ret,
				"WARNING: clGetProgramInfo(BINARIES)");
			free(buf);
			goto out;
		}

		//if ((_quirks & VG_OCL_AMD_BFI_INT) && !patched) {
		//	patched = ocl_amd_patch((unsigned char*)buf, szr);
		//	if (patched > 0) {
		//		clReleaseProgram(prog);
		//		goto rebuild;
		//	}
		//	fprintf(stderr,
		//		"WARNING: AMD BFI_INT patching failed\n");
		//	if (patched < 0) {
		//		/* Program was incompletely modified */
		//		free(buf);
		//		goto out;
		//	}
		//}

		kfp = fopen(bin_name, "wb");
		if (!kfp) {
			fprintf(stderr, "WARNING: "
				"could not save CL kernel binary: %s\n",
				strerror(errno));
		}
		else {
			sz = fwrite(buf, 1, szr, kfp);
			fclose(kfp);
			if (sz != szr) {
				fprintf(stderr,
					"WARNING: short write on CL kernel "
					"binary file: expected "
					"%zd, got %zd\n",
					szr, sz);
				_unlink(bin_name);
			}
		}
		free(buf);
	}

out:
	_program = prog;

	return 1;
}


/*Getting CRC32 hash of the program */
uint32_t OCLEngine::ocl_hash_program(const char* opts, const char* program, size_t size)
{
	const char* str;
	uint32_t h = 0;

	cl_platform_id pid = ocl_device_getplatform(_device_id);

	str = ocl_platform_getstr(pid, CL_PLATFORM_NAME);
	h = Utils::hash_crc32(h, str, strlen(str));

	str = ocl_platform_getstr(pid, CL_PLATFORM_VERSION);
	h = Utils::hash_crc32(h, str, strlen(str));

	str = ocl_device_getstr(_device_id, CL_DEVICE_NAME);
	h = Utils::hash_crc32(h, str, strlen(str));

	if (opts)
		h = Utils::hash_crc32(h, opts, strlen(opts));
	if (program && size) {
		h = Utils::hash_crc32(h, program, size);
	}
	return h;
}


/*Building a log of the program compilation process */
void OCLEngine::ocl_buildlog(cl_program prog)
{
	size_t logbufsize, logsize;
	char* log;
	int off = 0;
	cl_int ret;

	ret = clGetProgramBuildInfo(prog,
		_device_id,
		CL_PROGRAM_BUILD_LOG,
		0, nullptr,
		&logbufsize);
	if (ret != CL_SUCCESS) {
		ocl_error(ret, "clGetProgramBuildInfo");
		return;
	}

	log = (char*)malloc(logbufsize);
	if (!log) {
		fprintf(stderr, "Could not allocate build log buffer\n");
		return;
	}

	ret = clGetProgramBuildInfo(prog,
		_device_id,
		CL_PROGRAM_BUILD_LOG,
		logbufsize,
		log,
		&logsize);
	if (ret != CL_SUCCESS) {
		ocl_error(ret, "clGetProgramBuildInfo");

	}
	else {
		/* Remove leading newlines and trailing newlines/whitespace */
		log[logbufsize - 1] = '\0';
		for (off = logsize - 1; off >= 0; off--) {
			if ((log[off] != '\r') &&
				(log[off] != '\n') &&
				(log[off] != ' ') &&
				(log[off] != '\t') &&
				(log[off] != '\0'))
				break;
			log[off] = '\0';
		}
		for (off = 0; off < (int)logbufsize; off++) {
			if ((log[off] != '\r') &&
				(log[off] != '\n'))
				break;
		}

		fprintf(stderr, "Build log:\n%s\n", &log[off]);
	}
	free(log);
}


typedef struct {
	unsigned char e_ident[16];
	uint16_t e_type;
	uint16_t e_machine;
	uint32_t e_version;
	uint32_t e_entry;
	uint32_t e_phoff;
	uint32_t e_shoff;
	uint32_t e_flags;
	uint16_t e_ehsize;
	uint16_t e_phentsize;
	uint16_t e_phnum;
	uint16_t e_shentsize;
	uint16_t e_shnum;
	uint16_t e_shstrndx;
} vg_elf32_header_t;

typedef struct {
	uint32_t sh_name;
	uint32_t sh_type;
	uint32_t sh_flags;
	uint32_t sh_addr;
	uint32_t sh_offset;
	uint32_t sh_size;
	uint32_t sh_link;
	uint32_t sh_info;
	uint32_t sh_addralign;
	uint32_t sh_entsize;
} vg_elf32_shdr_t;

int OCLEngine::ocl_amd_patch_inner(unsigned char* binary, size_t size)
{
	vg_elf32_header_t* ehp;
	vg_elf32_shdr_t* shp, * nshp;
	uint32_t* instr;
	size_t off;
	int i, n, txt2idx, patched;

	ehp = (vg_elf32_header_t*)binary;
	if ((size < sizeof(*ehp)) ||
		memcmp(ehp->e_ident, "\x7f" "ELF\1\1\1\x64", 8) != 0 ||
		!ehp->e_shoff)
		return 0;

	off = ehp->e_shoff + (ehp->e_shstrndx * ehp->e_shentsize);
	nshp = (vg_elf32_shdr_t*)(binary + off);
	if ((off + sizeof(*nshp)) > size)
		return 0;

	shp = (vg_elf32_shdr_t*)(binary + ehp->e_shoff);
	n = 0;
	txt2idx = 0;
	for (i = 0; i < ehp->e_shnum; i++) {
		off = nshp->sh_offset + shp[i].sh_name;
		if (((off + 6) >= size) ||
			memcmp(binary + off, ".text", 6) != 0)
			continue;
		n++;
		if (n == 2)
			txt2idx = i;
	}
	if (n != 2)
		return 0;

	off = shp[txt2idx].sh_offset;
	instr = (uint32_t*)(binary + off);
	n = shp[txt2idx].sh_size / 4;
	patched = 0;
	for (i = 0; i < n; i += 2) {
		if (((instr[i] & 0x02001000) == 0) &&
			((instr[i + 1] & 0x9003f000) == 0x0001a000)) {
			instr[i + 1] ^= (0x0001a000 ^ 0x0000c000);
			patched++;
		}
	}

	return patched;
}

/*Patch AMD*/
int OCLEngine::ocl_amd_patch(unsigned char* binary, size_t size)
{
	vg_elf32_header_t* ehp;
	unsigned char* ptr;
	size_t offset = 1;
	int ninner = 0, nrun, npatched = 0;

	ehp = (vg_elf32_header_t*)binary;
	if ((size < sizeof(*ehp)) ||
		memcmp(ehp->e_ident, "\x7f" "ELF\1\1\1\0", 8) != 0 ||
		!ehp->e_shoff)
		return 0;

	offset = 1;
	while (offset < (size - 8)) {
		ptr = (unsigned char*)memchr(binary + offset,
			0x7f,
			size - offset);
		if (!ptr)
			return npatched;
		offset = ptr - binary;
		ehp = (vg_elf32_header_t*)ptr;
		if (((size - offset) < sizeof(*ehp)) ||
			memcmp(ehp->e_ident, "\x7f" "ELF\1\1\1\x64", 8) != 0 ||
			!ehp->e_shoff) {
			offset += 1;
			continue;
		}

		ninner++;
		nrun = ocl_amd_patch_inner(ptr, size - offset);
		npatched += nrun;
		npatched++;
		offset += 1;
	}
	return npatched;
}

/***********************************************************************
 * PROGRAM KERNEL
 ***********************************************************************/

 /*Function registration*/
int OCLEngine::ocl_kernel_create(int knum, const char* func)
{
	cl_kernel krn;
	cl_int ret;

	krn = clCreateKernel(_program, func, &ret);
	if (!krn) {
		fprintf(stderr, "clCreateKernel(%s)", func);
		ocl_error(ret, nullptr);
		clReleaseKernel(_kernel[knum]);
		_kernel[knum] = nullptr;
		return 0;
	}

	_kernel[knum] = krn;
	return 1;
}


static int ocl_arg_map[][8] = {
	/* hashes_out / found */
	{2, 0, -1},
	/* z_heap */
	{0, 1, 1, 0, 2, 2, -1},
	/* point_tmp */
	{0, 0, 2, 1, -1},
	/* row_in */
	{0, 2, -1},
	/* col_in */
	{0, 3, -1},
	/* target_table */
	//{2, 3, -1},

	/* bloom */
	{2, 3, -1},

	/* bloom */
	//    {2, 4, -1},
	/* hashes */
	//    {2, 5, -1},
	/* bits */
	//    {2, 6, -1},
};

/*Argument registration*/
int OCLEngine::ocl_kernel_arg_alloc(int arg, size_t size, int host)
{
	cl_mem clbuf;
	cl_int ret;
	int j, knum, karg;

	if (_arguments[arg]) {
		clReleaseMemObject(_arguments[arg]);
		_arguments[arg] = nullptr;
		_argument_size[arg] = 0;
	}

	clbuf = clCreateBuffer(_context, CL_MEM_READ_WRITE | (host ? CL_MEM_ALLOC_HOST_PTR : 0), size, nullptr, &ret);
	if (!clbuf) {
		fprintf(stderr, "clCreateBuffer(%d): ", arg);
		ocl_error(ret, nullptr);
		return 0;
	}

	clRetainMemObject(clbuf);
	_arguments[arg] = clbuf;
	_argument_size[arg] = size;

	for (j = 0; ocl_arg_map[arg][j] >= 0; j += 2) {
		knum = ocl_arg_map[arg][j];
		karg = ocl_arg_map[arg][j + 1];
		ret = clSetKernelArg(_kernel[knum], karg, sizeof(clbuf), &clbuf);
		if (ret) {
			fprintf(stderr, "clSetKernelArg(%d,%d): ", knum, karg);
			ocl_error(ret, nullptr);
			return 0;
		}
	}

	clReleaseMemObject(clbuf);
	return 1;
}


void* OCLEngine::ocl_map_arg_buffer(int arg, int rw)
{
	void* buf;
	cl_int ret;
	buf = clEnqueueMapBuffer(_command,
		_arguments[arg],
		CL_TRUE,
		(rw == 2) ? (CL_MAP_READ | CL_MAP_WRITE)
		: (rw ? CL_MAP_WRITE : CL_MAP_READ),
		0, _argument_size[arg],
		0, nullptr,
		nullptr,
		&ret);
	if (!buf) {
		fprintf(stderr, "clEnqueueMapBuffer(%d): ", arg);
		ocl_error(ret, nullptr);
		return nullptr;
	}
	return buf;
}


void OCLEngine::ocl_unmap_arg_buffer(int arg, void* buf)
{
	cl_int ret;
	cl_event ev;
	ret = clEnqueueUnmapMemObject(_command,
		_arguments[arg],
		buf,
		0, nullptr,
		&ev);
	if (ret != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueUnmapMemObject(%d): ", arg);
		ocl_error(ret, nullptr);
		return;
	}

	ret = clWaitForEvents(1, &ev);
	clReleaseEvent(ev);
	if (ret != CL_SUCCESS) {
		fprintf(stderr, "clWaitForEvent(clUnmapMemObject,%d): ", arg);
		ocl_error(ret, nullptr);
	}
}


int OCLEngine::ocl_kernel_int_arg(int kernel, int arg, int value)
{
	cl_int ret;
	ret = clSetKernelArg(_kernel[kernel],
		arg,
		sizeof(value),
		&value);
	if (ret) {
		fprintf(stderr, "clSetKernelArg(%d): ", arg);
		ocl_error(ret, nullptr);
		return 0;
	}
	return 1;
}

int OCLEngine::ocl_kernel_init()
{

	/*
	 * Function OpenCL - Setting starting points for matrix computation
	 * KERNEL ID : 0
	 * ec_add_grid(
	*              __global bn_word * points_out,
	*              __global bn_word * z_heap,
	*              __global bn_word *row_in,
	*              __global bignum *col_in
	 * )
	*
	*
	 * OpenCL Function - Computing Inversions
	 * KERNEL ID : 1
	 * heap_invert(
	*              __global bn_word *z_heap,
	*              int batch
	 * )
	*
	*
	 * OpenCL function - calculates hashes of points and searches for matches with hashes given in the structure of binary hashes
	 * KERNEL ID : 2
	 * hash_and_check(
	*              __global uint * found,          //The argument for writing the result of searching for matches of hashes of points in the list of binary hashes
	*              __global bn_word * points_in,   //XY array of Jacobi coordinates
	*              __global bn_word * z_heap,      //Array Z
	*              __global uint * tree            //Argument to store the structure of binary hashes
	*      )
	*
	*
	 * ARG values map:
	 * 0 = hash_and_check_bloom(found)
	 * 1 = ec_add_grid(z_heap), heap_invert(z_heap), hash_and_check(z_heap)
	 * 2 = ec_add_grid(points_out), hash_and_check(points_in)
	 * 3 = ec_add_grid(row_in)
	 * 4 = ec_add_grid(col_in)
	 * 5 = hash_and_check_bloom(bloom)
	 * 6 = hash_and_check_bloom(hashes)
	 * 7 = hash_and_check_bloom(bits)
	 */


	 //Connecting to OpenCL Script Functions
	if (!ocl_kernel_create(0, "ec_add_grid") ||
		!ocl_kernel_create(1, "heap_invert") ||
		!ocl_kernel_create(2, _addr_mode == 0 ? "hash_and_check_bloom_u" : (_addr_mode == 1 ? "hash_and_check_bloom_c" : "hash_and_check_bloom"))) {
		clReleaseProgram(_program);
		_program = nullptr;
		exit2("ocl_kernel_create", 1);
	}

	//Argument for writing the result of searching for matches of hashes of points in the list of binary hashes: hash_and_check (found)
	if (!ocl_kernel_arg_alloc(0, ARG_FOUND_SIZE, 1)) {
		exit2("ocl_kernel_arg_alloc", 1);
	}

	// Argument to store the structure of bloom data
	if (!ocl_kernel_arg_alloc(5, BLOOM_N, 0)) {
		exit2("ocl_kernel_arg_alloc", 1);
	}
	auto* bloomf = (unsigned char*)ocl_map_arg_buffer(5, 1);
	memcpy(bloomf, _bloom->get_bf(), BLOOM_N);
	ocl_unmap_arg_buffer(5, bloomf);

	//Argument to store the starting points for calculating the input matrix: ec_add_grid(col_in)
	if (!ocl_kernel_arg_alloc(4, 32 * 2 * _nrows, 1)) {
		printf("No memory ARG:4\n");
		exit2("ocl_kernel_arg_alloc", 1);
	}

	//z_heap & row_in
	if (!ocl_kernel_arg_alloc(1, round_up_pow2(32 * 2 * _round, 4096), 0) ||
		//ec_add_grid(z_heap), heap_invert(z_heap), hash_and_check(z_heap)
		!ocl_kernel_arg_alloc(2, round_up_pow2(32 * 2 * _round, 4096), 0) ||
		//ec_add_grid(points_out), hash_and_check(points_in)
		!ocl_kernel_arg_alloc(3, round_up_pow2(32 * 2 * _ncols, 4096), 1)) {  //ec_add_grid(row_in)
		printf("No memory ARG:1,2,3\n");
		exit2("ocl_kernel_arg_alloc", 1);
	}

	//Argument to store the size of the inversion queue: heap_invert(batch)
	if (!ocl_kernel_int_arg(1, 1, _invsize)) {
		exit2("ocl_kernel_int_arg", 1);
	}

	// bloom hashes
	if (!ocl_kernel_int_arg(2, 4, (int)_bloom->get_hashes())) {
		exit2("ocl_kernel_int_arg", 1);
	}
	// bloom bits
	if (!ocl_kernel_int_arg(2, 5, (int)_bloom->get_bits())) {
		exit2("ocl_kernel_int_arg", 1);
	}
	return 1;
}


int OCLEngine::ocl_kernel_start()
{

	cl_int ret;
	cl_event ev;
	size_t globalws[2] = { _ncols, _nrows };
	size_t invws = (_round) / _invsize;

	//Running the first function: ec_add_grid
	ret = clEnqueueNDRangeKernel(_command,
		_kernel[0],
		2,
		nullptr, globalws, nullptr,
		0, nullptr,
		&ev);
	if (ret != CL_SUCCESS) {
		ocl_error(ret, "clEnqueueNDRange(0)");
		return 0;
	}

	ret = clWaitForEvents(1, &ev);
	clReleaseEvent(ev);
	if (ret != CL_SUCCESS) {
		ocl_error(ret, "clWaitForEvents(NDRange,0)");
		return 0;
	}

	//Run the second function: heap_invert
	ret = clEnqueueNDRangeKernel(_command,
		_kernel[1],
		1,
		nullptr, &invws, nullptr,
		0, nullptr,
		&ev);
	if (ret != CL_SUCCESS) {
		ocl_error(ret, "clEnqueueNDRange(1)");
		return 0;
	}

	ret = clWaitForEvents(1, &ev);
	clReleaseEvent(ev);
	if (ret != CL_SUCCESS) {
		ocl_error(ret, "clWaitForEvents(NDRange,1)");
		return 0;
	}


	//Running the third function: hash_and_check
	ret = clEnqueueNDRangeKernel(_command,
		_kernel[2],
		2,
		nullptr, globalws, nullptr,
		0, nullptr,
		&ev);
	if (ret != CL_SUCCESS) {
		ocl_error(ret, "clEnqueueNDRange(2)");
		return 0;
	}

	ret = clWaitForEvents(1, &ev);
	clReleaseEvent(ev);
	if (ret != CL_SUCCESS) {
		ocl_error(ret, "clWaitForEvents(NDRange,2)");
		return 0;
	}

	return 1;
}


/***********************************************************************
 * POINT <--> RAW
 ***********************************************************************/

static void ocl_get_bignum_raw(BIGNUM* bn, const unsigned char* buf)
{
	bn_expand(bn, 256);
	memcpy(bn->d, buf, 32);
	bn->top = (32 / sizeof(BN_ULONG));
}

static void ocl_put_bignum_raw(unsigned char* buf, const BIGNUM* bn)
{
	auto bnlen = (bn->top * sizeof(BN_ULONG));
	if (bnlen >= 32) {
		memcpy(buf, bn->d, 32);
	}
	else {
		memcpy(buf, bn->d, bnlen);
		memset(buf + bnlen, 0, 32 - bnlen);
	}
}

struct ec_point_st {
	const EC_METHOD* meth;
	BIGNUM X;
	BIGNUM Y;
	BIGNUM Z;
	int Z_is_one;
};

void OCLEngine::ocl_get_point(EC_POINT* ppnt, const unsigned char* buf)
{
	static const unsigned char mont_one[] = { 0x01, 0x00, 0x00, 0x03, 0xd1 };
	ocl_get_bignum_raw(&ppnt->X, buf);
	ocl_get_bignum_raw(&ppnt->Y, buf + 32);
	if (!ppnt->Z_is_one) {
		ppnt->Z_is_one = 1;
		BN_bin2bn(mont_one, sizeof(mont_one), &ppnt->Z);
	}
}

void OCLEngine::ocl_put_point(unsigned char* buf, const EC_POINT* ppnt)
{
	assert(ppnt->Z_is_one);
	ocl_put_bignum_raw(buf, &ppnt->X);
	ocl_put_bignum_raw(buf + 32, &ppnt->Y);
}


void OCLEngine::ocl_put_point_tpa(unsigned char* buf, int cell, const EC_POINT* ppnt)
{
	unsigned char pntbuf[64];
	int start, i;

	ocl_put_point(pntbuf, ppnt);

	start = ((((2 * cell) / ACCESS_STRIDE) * ACCESS_BUNDLE) +
		(cell % (ACCESS_STRIDE / 2)));
	for (i = 0; i < 8; i++) {
		memcpy(buf + 4 * (start + i * ACCESS_STRIDE),
			pntbuf + (i * 4),
			4);
	}
	for (i = 0; i < 8; i++) {
		memcpy(buf + 4 * (start + (ACCESS_STRIDE / 2) + (i * ACCESS_STRIDE)),
			pntbuf + 32 + (i * 4),
			4);
	}
}

void OCLEngine::ocl_get_point_tpa(EC_POINT* ppnt, const unsigned char* buf, int cell)
{
	unsigned char pntbuf[64];
	int start, i;

	start = ((((2 * cell) / ACCESS_STRIDE) * ACCESS_BUNDLE) +
		(cell % (ACCESS_STRIDE / 2)));
	for (i = 0; i < 8; i++) {
		memcpy(pntbuf + (i * 4),
			buf + 4 * (start + i * ACCESS_STRIDE),
			4);
	}
	for (i = 0; i < 8; i++) {
		memcpy(pntbuf + 32 + (i * 4),
			buf + 4 * (start + (ACCESS_STRIDE / 2) + (i * ACCESS_STRIDE)),
			4);
	}

	ocl_get_point(ppnt, pntbuf);
}


double OCLEngine::time_diff(struct timeval x, struct timeval y)
{
	double x_ms, y_ms, diff;
	x_ms = (double)x.tv_sec * 1000000 + (double)x.tv_usec;
	y_ms = (double)y.tv_sec * 1000000 + (double)y.tv_usec;
	diff = (double)y_ms - (double)x_ms;
	return diff;
}

int OCLEngine::check_hash_binary(const uint8_t* hash)
{
	uint8_t* temp_read;
	uint64_t half, min, max, current; //, current_offset
	int64_t rcmp;
	int32_t r = 0;
	min = 0;
	current = 0;
	max = DATA_SIZE / 20;
	half = DATA_SIZE / 20;
	while (!r && half >= 1) {
		half = (max - min) / 2;
		temp_read = DATA + ((current + half) * 20);
		rcmp = memcmp(hash, temp_read, 20);
		if (rcmp == 0) {
			r = 1;  //Found!!
		}
		else {
			if (rcmp < 0) { //data < temp_read
				max = (max - half);
			}
			else { // data > temp_read
				min = (min + half);
			}
			current = min;
		}
	}
	return r;
}



