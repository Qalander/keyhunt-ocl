#include <iostream>
#include "oclengine.h"
#include "argparse.h"

bool should_exit = false;

BOOL WINAPI CtrlHandler(DWORD fdwCtrlType)
{
    switch (fdwCtrlType) {
    case CTRL_C_EVENT:
        printf("\n\nCtrl-C event\n\n");
        should_exit = true;
        return TRUE;

    default:
        return TRUE;
    }
}

int main(int argc, const char *argv[])
{

    std::string clfilename = "gpu.cl";
    std::string bin_file   = "";
    std::string pkey_base  = "";
    int32_t platform_id    = 0;
    int32_t device_id      = 0;
    int32_t addr_mode      = 0;
    int32_t unlim_round    = 0;
    uint32_t nrows         = 0;
    uint32_t ncols         = 0;
    uint32_t invsize       = 0;

    argparse::ArgumentParser parser("keyhunt-ocl", "hunt for bitcoin private keys.");

    parser.add_argument("-p", "--platform", "Platform id [default: 0]",                                            false);
    parser.add_argument("-d", "--device",   "Device id [default: 0]",                                              false);
    parser.add_argument("-r", "--rows",     "Grid rows [default: 0(auto)]",                                        false);
    parser.add_argument("-c", "--cols",     "Grid cols [default: 0(auto)]",                                        false);
    parser.add_argument("-i", "--invsize",  "Mod inverse batch size [default: 0(auto)]",                           false);
    parser.add_argument("-m", "--mode",     "Address mode [default: 0] [0: uncompressed, 1: compressed, 2: both]", true);
    parser.add_argument("-u", "--unlim",    "Unlimited rounds [default: 0] [0: false, 1: true]",                   false);
    parser.add_argument("-k", "--privkey",  "Base privkey",                                                        false);
    parser.add_argument("-f", "--file",     "RMD160 Address binary file path",                                     true);
    parser.enable_help();

    auto err = parser.parse(argc, argv);
    if (err) {
        std::cout << err << std::endl;
        parser.print_help();
        return -1;
    }

    if (parser.exists("help")) {
        parser.print_help();
        return 0;
    }

    if (parser.exists("platform"))
        platform_id = parser.get<int32_t>("p");

    if (parser.exists("device"))
        device_id = parser.get<int32_t>("d");

    if (parser.exists("rows"))
        nrows = parser.get<uint32_t>("r");

    if (parser.exists("cols"))
        ncols = parser.get<uint32_t>("c");

    if (parser.exists("invsize"))
        invsize = parser.get<uint32_t>("i");

    if (parser.exists("mode"))
        addr_mode = parser.get<int32_t>("m");

    if (parser.exists("unlim"))
        unlim_round = parser.get<int32_t>("u");

    if (parser.exists("privkey"))
        pkey_base = parser.get<std::string>("k");

    if (parser.exists("file"))
        bin_file = parser.get<std::string>("f");

    if (addr_mode > 2 || addr_mode < 0) {
        std::cout << "invalid address mode: " << addr_mode << std::endl;
        return -1;
    }

    std::cout << "\n" << "ARGUMENTS:" << std::endl;
    std::cout << "\tPLATFORM ID: " << platform_id << "[default: 0]" << std::endl;
    std::cout << "\tDEVICE ID  : " << device_id << "[default: 0]" << std::endl;
    std::cout << "\tNUM ROWS   : " << nrows << "[default: 0(auto)]" << std::endl;
    std::cout << "\tNUM COLS   : " << ncols << "[default: 0(auto)]" << std::endl;
    std::cout << "\tINVSIZE    : " << invsize << "[default: 0(auto)]" << std::endl;
    std::cout << "\tADDR_MODE  : " << addr_mode << "[0: uncompressed, 1: compressed, 2: both]" << std::endl;
    std::cout << "\tUNLIM ROUND: " << unlim_round << std::endl;
    std::cout << "\tPKEY BASE  : " << pkey_base << std::endl;
    std::cout << "\tBIN FILE   : " << bin_file << std::endl << std::endl;

    if (SetConsoleCtrlHandler(CtrlHandler, TRUE)) {
        OCLEngine *ocl = new OCLEngine(platform_id, device_id, clfilename.c_str(), ncols, nrows,
                                       invsize, unlim_round, addr_mode, pkey_base.c_str(), bin_file.c_str(), should_exit);
        if (ocl->is_ready()) {
            ocl->loop(should_exit);
        }
        delete ocl;
        return 0;
    } else {
        printf("error: could not set control-c handler\n");
        return 1;
    }

    return 0;
}
