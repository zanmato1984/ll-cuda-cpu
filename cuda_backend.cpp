#include <cuda.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/Instrumentation.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <sstream>
#include <unordered_set>

using namespace llvm;

#define CHECK_CUDA(call, msg)                                                  \
  if (auto res = (call); res != CUDA_SUCCESS)                                  \
  throw std::runtime_error(msg + std::string(" failed: ") + std::to_string(res))

// CUDA runtime function declares.
const char *cuda_runtime_function_decls =
    "\ndeclare void @loop_cuda(float*, float*, i64);\n";

void runCuda(LLVMContext &context, Module *module, Function *entry_func,
             std::vector<float> &in, std::vector<float> &out, size_t n) {
  {
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();
  }

  /// Massage the entry function by setting kernel metadata, removing the CPU
  /// `loop_cuda` stub. Then obtain LLIR.
  std::string llir;
  {
    llvm::NamedMDNode *md =
        module->getOrInsertNamedMetadata("nvvm.annotations");
    llvm::Metadata *md_vals[] = {
        llvm::ConstantAsMetadata::get(entry_func),
        llvm::MDString::get(context, "kernel"),
        llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 1))};
    md->addOperand(llvm::MDNode::get(context, md_vals));
    std::unordered_set<llvm::Function *> roots{entry_func};
    std::vector<llvm::Function *> rt_funcs;
    for (auto &f : *module) {
      if (roots.count(&f)) {
        continue;
      }
      rt_funcs.push_back(&f);
    }
    for (auto &f : rt_funcs) {
      f->removeFromParent();
    }

    std::stringstream ss;
    llvm::raw_os_ostream os(ss);
    module->print(os, nullptr);
    os.flush();

    // TODO: Don't know what below is doing.
    for (auto &f : rt_funcs) {
      module->getFunctionList().push_back(f);
    }
    module->eraseNamedMetadata(md);

    llir = cuda_runtime_function_decls + ss.str();
  }

  /// Parse LLIR and get CUDA module.
  std::unique_ptr<Module> cuda_module;
  {
    auto mem_buff = llvm::MemoryBuffer::getMemBuffer(llir, "", false);
    llvm::SMDiagnostic diag;
    cuda_module = llvm::parseIR(mem_buff->getMemBufferRef(), diag, context);
    if (!cuda_module)
      throw std::runtime_error("CUDA IR parse error: " +
                               diag.getMessage().str());
  }

  /// Create CUDA target machine.
  std::unique_ptr<llvm::TargetMachine> cuda_machine;
  {
    std::string err;
    auto cuda_target = llvm::TargetRegistry::lookupTarget("nvptx64", err);
    if (!cuda_target)
      throw std::runtime_error("Couldn't find nvptx64 target: " + err);

    cuda_machine =
        std::unique_ptr<llvm::TargetMachine>(cuda_target->createTargetMachine(
            "nvptx64-nvidia-cuda", "sm_30", "", llvm::TargetOptions(),
            llvm::Reloc::Static));
  }

  /// Emit PTX of LLIR.
  llvm::SmallString<256> ptx;
  {
    llvm::raw_svector_ostream formatted_os(ptx);
    llvm::legacy::PassManager ptxgen_pm;
    cuda_module->setDataLayout(cuda_machine->createDataLayout());
    if (cuda_machine->addPassesToEmitFile(ptxgen_pm, formatted_os, nullptr,
                                          TargetMachine::CGFT_AssemblyFile,
                                          true))
      throw std::runtime_error("Generate PTX failed");
    ptxgen_pm.run(*cuda_module);
  }

  /// Init CUDA runtime and obtain some hardware attributes.
  int block_size = 0, grid_size = 0, shared_mem_size = 0;
  {
    CUcontext cu_context;
    CUdevice cu_device;
    CHECK_CUDA(cuInit(0), "CUDA init");
    CHECK_CUDA(cuDeviceGet(&cu_device, 0), "CUDA device get");
    CHECK_CUDA(cuCtxCreate(&cu_context, 0, cu_device), "CUDA context creation");
    CHECK_CUDA(cuDeviceGetAttribute(&block_size,
                                    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                    cu_device),
               "CUDA device get block size");
    CHECK_CUDA(cuDeviceGetAttribute(&grid_size,
                                    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                    cu_device),
               "CUDA device get grid size");
    CHECK_CUDA(cuDeviceGetAttribute(
                   &shared_mem_size,
                   CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cu_device),
               "CUDA device get shared mem size");
    CHECK_CUDA(cuCtxSetCurrent(cu_context), "CUDA ctx set current");
  }

  /// Link CUDA module with CUDA runtime library, and obtain cubin.
  void *cubin{nullptr};
  size_t cubin_size{0};
  std::vector<CUjit_option> option_keys{CU_JIT_LOG_VERBOSE,
                                        CU_JIT_THREADS_PER_BLOCK};
  std::vector<void *> option_values{reinterpret_cast<void *>(1),
                                    reinterpret_cast<void *>(block_size)};
  unsigned num_options = option_keys.size();
  {
    CUlinkState link_state;
    CHECK_CUDA(cuLinkCreate(num_options, &option_keys[0], &option_values[0],
                            &link_state),
               "CUDA link creation");
    CHECK_CUDA(cuLinkAddFile(link_state, CU_JIT_INPUT_LIBRARY,
                             "cuda_runtime_functions.a", num_options,
                             &option_keys[0], &option_values[0]),
               "CUDA link add file");
    CHECK_CUDA(
        cuLinkAddData(link_state, CU_JIT_INPUT_PTX,
                      static_cast<void *>(const_cast<char *>(ptx.data())),
                      ptx.size() + 1, nullptr, num_options, &option_keys[0],
                      &option_values[0]),
        "CUDA link add data");
    CHECK_CUDA(cuLinkComplete(link_state, &cubin, &cubin_size), "CUDA link");
    if (!cubin || cubin_size == 0)
      throw std::runtime_error("CUDA bin invalid");
  }

  /// Create CUDA module and obtain runnable kernel.
  CUmodule cu_module;
  CUfunction cu_kernel;
  {
    CHECK_CUDA(cuModuleLoadDataEx(&cu_module, cubin, num_options,
                                  option_keys.data(), option_values.data()),
               "CUDA module load");
    CHECK_CUDA(cuModuleGetFunction(&cu_kernel, cu_module,
                                   entry_func->getName().data()),
               "CUDA get kernel");
  }

  /// Call kernel with parameters.
  {
    CUdeviceptr d_in = 0, d_out = 0;
    CHECK_CUDA(cuMemAlloc(&d_in, n * sizeof(float)), "CUDA malloc in");
    CHECK_CUDA(cuMemAlloc(&d_out, n * sizeof(float)), "CUDA malloc out");
    CHECK_CUDA(cuMemcpyHtoD(d_in, in.data(), n * sizeof(float)),
               "CUDA memcpy in");
    std::vector<void *> params{&d_in, &d_out, &n};
    CHECK_CUDA(cuLaunchKernel(cu_kernel, grid_size, 1, 1, block_size, 1, 1,
                              shared_mem_size, nullptr, params.data(), nullptr),
               "CUDA launch kernel");
    CHECK_CUDA(cuMemcpyDtoH(out.data(), d_out, n * sizeof(float)),
               "CUDA memcpy out");
    CHECK_CUDA(cuMemFree(d_in), "CUDA free in");
    CHECK_CUDA(cuMemFree(d_out), "CUDA free out");
  }
}
