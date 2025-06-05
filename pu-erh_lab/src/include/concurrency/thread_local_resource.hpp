// thread_local_resource.hpp
#pragma once
#include <functional>
#include <memory>
#include <mutex>

// 可泛化给任何类型 T
template <typename T>
class ThreadLocalResource {
 public:
  using Initializer = std::function<std::unique_ptr<T>()>;

  static void SetInitializer(Initializer init) { GetInitFunc() = std::move(init); }

  static T&   Get() {
    thread_local std::unique_ptr<T> instance = GetInitFunc()();
    return *instance;
  }

 private:
  static Initializer& GetInitFunc() {
    static Initializer init_func;
    return init_func;
  }
};