#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/Cloning.h>

using namespace llvm;

void runCpu(LLVMContext &context, Module *module, Function *entry_func,
            std::vector<float> &in, std::vector<float> &out, size_t n) {
  {
    InitializeNativeTarget();
    InitializeAllTargetMCs();
    InitializeNativeTargetAsmParser();
    InitializeNativeTargetAsmPrinter();
  }

  /// Build the execution engine.
  ExecutionEngine *engine;
  {
    std::string err;
    std::unique_ptr<llvm::Module> owner(module);
    llvm::EngineBuilder engine_builder(std::move(owner));
    engine_builder.setErrorStr(&err);
    engine_builder.setEngineKind(llvm::EngineKind::JIT);
    llvm::TargetOptions to;
    to.EnableFastISel = true;
    engine_builder.setTargetOptions(to);
    engine = engine_builder.create();
    if (!engine) {
      throw std::runtime_error("Execution engine creation error: " + err);
    }
  }

  /// Execute code.
  {
    engine->finalizeObject();
    using entry_func_t = void (*)(float *, float *, size_t);
    auto entry_func_native = reinterpret_cast<entry_func_t>(
        engine->getFunctionAddress(entry_func->getName()));
    entry_func_native(in.data(), out.data(), n);
  }
}