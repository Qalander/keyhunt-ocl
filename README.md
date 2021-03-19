# keyhunt-ocl
_Hunt for Bitcoin private keys._

This is a modified version of [oclexplorer](https://github.com/svtrostov/oclexplorer.git) by svtrostov.

To convert Bitcoin legacy addresses to RIPEMD160 hasehs use this [b58dec](https://github.com/kanhavishva/b58dec).

It is important to binary sort the RIPEMD160 file before giving it to the program, otherwise binary search function would not work properly. To do this work use this [RMD160-Sort](https://github.com/kanhavishva/RMD160-Sort).


## Changes

- Renamed from oclexplorer to KeyHunt (inspired from [keyhunt](https://github.com/albertobsd/keyhunt) by albertobsd).
- Modified opencl kernel to match for both compressed and un-compressed keys addresses.
- Added bloom filter for fast matchings.
- Transfer only bloom data to GPU device and keep hash160 data in system memory, this way we can load a very large hash file.
- For args parsing it uses [argparse](https://github.com/jamolnng/argparse) by jamolnng)
- It supports GPU only.

## ToDo

- Add feature to search in given key-space range.

## Usage

```
λ keyhunt-ocl.exe -h
Usage: keyhunt-ocl [options...]
Options:
    -p, --platform         Platform id [default: 0]
    -d, --device           Device id [default: 0]
    -r, --rows             Grid rows [default: 0(auto)]
    -c, --cols             Grid cols [default: 0(auto)]
    -i, --invsize          Mod inverse batch size [default: 0(auto)]
    -m, --mode             Address mode [default: 0] [0: uncompressed, 1: compressed, 2: both] (Required)
    -u, --unlim            Unlimited rounds [default: 0] [0: false, 1: true]
    -k, --privkey          Base privkey
    -f, --file             RMD160 Address binary file path (Required)
    -h, --help             Shows this page
```

```
keyhunt-ocl.exe -m 1 -f G:/BTCADDRESSES/address6-160-sorted.bin

ARGUMENTS:
        PLATFORM ID: 0[default: 0]
        DEVICE ID  : 0[default: 0]
        NUM ROWS   : 0[default: 0(auto)]
        NUM COLS   : 0[default: 0(auto)]
        INVSIZE    : 0[default: 0(auto)]
        ADDR_MODE  : 1[0: uncompressed, 1: compressed, 2: both]
        UNLIM ROUND: 0
        PKEY BASE  :
        BIN FILE   : G:/BTCADDRESSES/address6-160-sorted.bin

Loading addresses: 100 %
Loaded addresses : 195855350 in 171.433410 sec

Bloom at 000001B68EAF0430
        Version    : 2.1
        Entries    : 391710700
        Error      : 0.0000100000
        Bits       : 9386424816
        Bits/Elem  : 23.962646
        Bytes      : 1173303102 (1118 MB)
        Hash funcs : 17

Loading program file: gpu.cl


MATRIX:
        Grid size  : 3584x4096
        Total      : 14680064
        Mod inverse: 14336 threads [1024 ops/thread]

DEVICE INFO:
        Device              : GeForce GTX 1650
        Vendor              : NVIDIA Corporation (10de)
        Driver              : 461.40
        Profile             : FULL_PROFILE
        Version             : OpenCL 1.2 CUDA
        Max compute units   : 14
        Max workgroup size  : 1024
        Global memory       : 4294967296
        Max allocation      : 1073741824


Iteration 1 at [2021-03-20 00:10:01] from: af34b0f60ded1c5597e31d5ed65a27755f3621e7ed3b3ecb3151fb9fbada2dcd
[af34b0f60ded1c5597e31d5ed65a27755f3621e7ed3b3ecb3151fba04c1a2dcd] [round 167: 0.21s (69.28 Mkey/s)] [total 2,451,570,688 (65.94 Mkey/s)]

Ctrl-C event

```
    
## Building

- Microsoft Visual Studio Community 2019 
- OpenCL 1.2

## License
keyhunt-ocl is licensed under GPLv3.
    
