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
    
## Building

- Microsoft Visual Studio Community 2019 
- OpenCL 1.2

## License
keyhunt-ocl is licensed under GPLv3.
    
