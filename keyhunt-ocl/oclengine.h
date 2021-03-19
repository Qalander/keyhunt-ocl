#ifndef OCLENGINE_H
#define OCLENGINE_H

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>

#include <CL/cl.h>

#include <openssl/ec.h>
#include <openssl/bn.h>
#include <openssl/rand.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>

#include "bloom.h"
#include "utils.h"

#include <string>

/***********************************************************************
 * Definitions and constants
 ***********************************************************************/

/*Working with bits*/
#define BIT_SET(a, b) ((a) |= (1<<(b)))
#define BIT_CLEAR(a, b) ((a) &= ~(1<<(b)))
#define BIT_FLIP(a, b) ((a) ^= (1<<(b)))
#define BIT_CHECK(a, b) ((a) & (1<<(b)))

#define MAX_KERNEL 3
#define MAX_ARG 8

#define is_pow2(v) (!((v) & ((v)-1)))
#define round_up_pow2(x, a) (((x) + ((a)-1)) & ~((a)-1))

#define ACCESS_BUNDLE 1024
#define ACCESS_STRIDE (ACCESS_BUNDLE/8)

#define ARG_FOUND_SIZE (24*2)

class OCLEngine
{
public:
    /***********************************************************************
     * OCLEngine
     ***********************************************************************/
    OCLEngine(int platform_id, int device_id, const char *program, uint32_t ncols,
              uint32_t nrows, uint32_t invsize, bool is_unlim_round, int32_t addr_mode, const char *pkey_base,
              const char *filename, bool &should_exit);
    ~OCLEngine();

    static void exit2(const char *err, int ret);
    bool is_ready() const;

    void loop(bool &should_exit);

private:
    /***********************************************************************
     * OpenCL debugging and support
     ***********************************************************************/
    static const char *ocl_strerror(cl_int ret);
    static void        ocl_error(int code, const char *desc);
    void               ocl_print_info();

    /***********************************************************************
     * PLATFORM
     ***********************************************************************/
    static cl_platform_id  ocl_platform_get(int num);
    static int             ocl_platform_list(cl_platform_id **list_out);
    static void            ocl_platforms_info(cl_platform_id *ids, int np, int base);
    static const char     *ocl_platform_getstr(cl_platform_id pid, cl_platform_info param);

    /***********************************************************************
     * DEVICE
     ***********************************************************************/
    static cl_device_id    ocl_device_manual(int platformidx, int deviceidx);
    static cl_device_id    ocl_device_get(cl_platform_id pid, int num);
    static int             ocl_devices_list(cl_platform_id pid, cl_device_id **list_out);
    static void            ocl_devices_info(cl_platform_id pid, cl_device_id *ids, int nd, int base);
    static cl_platform_id  ocl_device_getplatform(cl_device_id did);
    static cl_device_type  ocl_device_gettype(cl_device_id did);
    static const char     *ocl_device_getstr(cl_device_id did, cl_device_info param);
    static size_t          ocl_device_getsizet(cl_device_id did, cl_device_info param);
    static cl_ulong        ocl_device_getulong(cl_device_id did, cl_device_info param);
    static cl_uint         ocl_device_getuint(cl_device_id did, cl_device_info param);

    /***********************************************************************
    * PROGRAM
    ***********************************************************************/
    static unsigned int ocl_get_quirks(cl_device_id did, char *optbuf);
    static void         ocl_get_quirks_str(unsigned int quirks, char *optbuf);
    int                 ocl_load_program(const char *filename, const char *opts);
    uint32_t            ocl_hash_program(const char *opts, const char *program, size_t size);
    void                ocl_buildlog(cl_program prog);
    static int          ocl_amd_patch_inner(unsigned char *binary, size_t size);
    static int          ocl_amd_patch(unsigned char *binary, size_t size);

    /***********************************************************************
    * PROGRAM KERNEL
    ***********************************************************************/
    int   ocl_kernel_create(int knum, const char *func);
    int   ocl_kernel_arg_alloc(int arg, size_t size, int host);
    void *ocl_map_arg_buffer(int arg, int rw);
    void  ocl_unmap_arg_buffer(int arg, void *buf);
    int   ocl_kernel_int_arg(int kernel, int arg, int value);
    int   ocl_kernel_init();
    int   ocl_kernel_start();

    /***********************************************************************
    * POINT <--> RAW
    ***********************************************************************/
    static void ocl_get_point(EC_POINT *ppnt, const unsigned char *buf);
    static void ocl_put_point(unsigned char *buf, const EC_POINT *ppnt);
    static void ocl_put_point_tpa(unsigned char *buf, int cell, const EC_POINT *ppnt);
    static void ocl_get_point_tpa(EC_POINT *ppnt, const unsigned char *buf, int cell);

    /***********************************************************************
    * BINARY CHECK
    ***********************************************************************/
    int check_hash_binary(const uint8_t *hash);

    /***********************************************************************
    * TIME
    ***********************************************************************/
    static double time_diff(struct timeval x, struct timeval y);

    /***********************************************************************
    * REPORT
    ***********************************************************************/
    static void report(BIGNUM* bn_tmp, const BIGNUM* bn_key, KeyInfo* info, uint32_t found_delta, const uint8_t* found_hash,
                       uint8_t* hash_buf, time_t* now, char* time_buf, char* buffer, char* tmp, uint8_t* pkey_s, FILE* ffd, PubType pubtype);
    std::string formatThousands(uint64_t x);

private:
    Bloom              *_bloom;                  //Bloom filter
    cl_platform_id      _platform_id;            //Platform
    cl_device_id        _device_id;              //Device
    cl_context          _context;                //Context
    cl_command_queue    _command;                //Command
    cl_program          _program;                //Program
    uint64_t            _ncols;                  //Number of columns in a matrix
    uint64_t            _nrows;                  //Number of rows in a matrix
    int32_t             _addr_mode;              //Address mode
    bool                _is_unlim_round;         //A sign indicating that there should be an unlimited number of rounds, i.e. search from a specific key to victory
    uint64_t            _round;                  //Total number of matrix elements
    uint64_t            _invsize;                //Queue size for mod inverse

    uint64_t            _quirks;                 //Compiler options
    cl_kernel           _kernel[MAX_KERNEL];     //External CL program functions on the device
    cl_mem              _arguments[MAX_ARG];     //Function arguments
    size_t              _argument_size[MAX_ARG]; //Size of arguments

    const char        *_pkey_base;               //Initial private key
    uint64_t            BLOOM_N;
    uint64_t            DATA_SIZE;
    uint8_t            *DATA;
    bool                READY;
};

#endif // OCLENGINE_H
