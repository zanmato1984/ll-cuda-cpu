#include <iostream>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/Instrumentation.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/Cloning.h>

using namespace llvm;

enum Target {
  CPU,
  CUDA,
};

#ifdef HAVE_CUDA
void runCuda(LLVMContext &context, Module *module, Function *entry_func,
             std::vector<float> &in, std::vector<float> &out, size_t n);
#endif
void runCpu(LLVMContext &context, Module *module, Function *entry_func,
            std::vector<float> &in, std::vector<float> &out, size_t n);

constexpr auto default_target = CUDA;
constexpr size_t default_n = 1024;

int main(int argc, char *argv[]) {
  size_t n = default_n;
  auto target = default_target;

  {
    if (argc >= 2) {
      if (std::string(argv[1]) == "cpu") {
        target = CPU;
      }
    }
    if (argc == 3)
      n = std::atol(argv[2]);

#ifndef HAVE_CUDA
    if (target == CUDA) {
      throw std::runtime_error("CUDA not enabled");
    }
#endif
  }

  LLVMContext context;
  Module *module;
  Function *entry_func;
  std::vector<float> in(n, 256.0), out(n, 0);

  /// Load runtime functions.
  {
    auto buffer_or_error =
        llvm::MemoryBuffer::getFile("cpu_runtime_functions.bc");
    if (buffer_or_error.getError())
      throw std::runtime_error("Runtime function load error: " +
                               buffer_or_error.getError().message());

    llvm::MemoryBuffer *buffer = buffer_or_error.get().get();
    auto owner = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
    if (owner.takeError())
      throw std::runtime_error("Runtime function bit code parse error");

    module = owner.get().release();
    if (!module)
      throw std::runtime_error("Runtime module empty");
  }

  /// Generate IR for entry:
  {
    // The generated function is like:
    // void entry(float * in, float * out, size_t n) {
    //     loop_(cpu/cuda)(in, out, n);
    // }
    entry_func =
        Function::Create(FunctionType::get(Type::getVoidTy(context),
                                           {Type::getFloatPtrTy(context),
                                            Type::getFloatPtrTy(context),
                                            Type::getInt64Ty(context)},
                                           false),
                         Function::ExternalLinkage, "entry", module);
    BasicBlock *entry_bb = BasicBlock::Create(context, "", entry_func);
    IRBuilder<> builder(entry_bb);

    Argument *arg_in = &*entry_func->arg_begin();
    arg_in->setName("in");
    Argument *arg_out = &*entry_func->arg_begin() + 1;
    arg_out->setName("out");
    Argument *arg_n = &*entry_func->arg_begin() + 2;
    arg_n->setName("n");

    // Note that for `loop_cuda`, it is now the CPU stub (in
    // cpu_runtime_functions), will be replaced with the CUDA impl (in
    // cuda_runtime_functions) later.
    auto loop_func =
        module->getFunction(target == CUDA ? "loop_cuda" : "loop_cpu");

    builder.CreateCall(loop_func, {arg_in, arg_out, arg_n});
    builder.CreateRetVoid();
    verifyFunction(*entry_func, &llvm::errs());
  }

  /// Execute the code.
  {
#ifdef HAVE_CUDA
    if (target == CUDA) {
      runCuda(context, module, entry_func, in, out, n);
#else
    if (false) {
#endif
    } else if (target == CPU)
      runCpu(context, module, entry_func, in, out, n);
  }

  /// Output.
  {
    std::cout << "Result: "
              << "\n";
    for (auto r : out) {
      std::cout << r << ",";
    }
  }

  llvm_shutdown();

  return 0;
}
