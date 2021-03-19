#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <ctime>
#include <openssl/bn.h>
#include <openssl/ec.h>

#include "winglue.h"

typedef enum PubType {
	UNCOMPRESSED = 0,
	COMPRESSED
} PubType;

typedef struct KeyInfo {
	uint8_t private_bin[32];
	uint8_t private_hex[65];
	uint8_t private_wif[65];
	uint8_t public_x[32];
	uint8_t public_y[32];
	uint8_t publicu_bin[65];
	uint8_t publicu_hex[132];
	uint8_t publicc_bin[33];
	uint8_t publicc_hex[68];
	uint8_t public_sha256_bin[32];
	uint8_t public_sha256_hex[65];
	uint8_t public_ripemd160_bin[20];
	uint8_t public_ripemd160_hex[41];
	uint8_t address_hex[64];
} KeyInfo;

typedef struct HashRate {
	struct timeval time_start;
	struct timeval time_now;
	double runtime;
	double hashrate;
	const char* unit;
} HashRate;


class Utils {
public:
	Utils();

	static uint32_t hash_crc32(uint32_t crc32_start, const void* buf, size_t n);

	static uint8_t* bin2hex(uint8_t* buf, const uint8_t* from, size_t n);

	static int hex2bin(uint8_t* buf, const uint8_t* from, size_t n);

	static double time_diff(struct timeval x, struct timeval y);

	static void b58_encode_check(void* buf, size_t len, char* result);

	static int b58_decode_check(const char* input, void* buf, size_t len);

	static void encode_privkey(const BIGNUM* bn, int addrtype, uint8_t* bin_result, uint8_t* wit_result);

	static KeyInfo* get_key_info(const BIGNUM* private_key, PubType type);

	static int set_pkey(const BIGNUM* bnpriv, EC_KEY* pkey);

	static void hashrate_update(HashRate* hr, uint64_t value);

};

#endif // UTILS_H
