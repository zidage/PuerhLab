# Why Another `third_party` Folder

Because lensfun has not updated their Release for a long time, and for the newest version, we have to build it from source every time, so we decided to include the source code of lensfun in our project. We will update the source code of lensfun when there is a new release, and we will also update the source code of lensfun if there are some critical bugs that need to be fixed.

## Why We Need A 'CMakeLists.txt' File in `third_party`

Because we need to build the source code of lensfun, and on Windows machines, we need to reconfigure the CMakeLists.txt file from the original one to make it compatible with the Windows environment. We will also update the CMakeLists.txt file when there is a new release of lensfun, and we will also update the CMakeLists.txt file if there are some critical bugs that need to be fixed.

## How about the `metal-cpp` Folder?

Since Pu-erh Lab v0.2.0, a new Metal support has been added, and we need to include the source code of metal-cpp in our project. We will update the source code of metal-cpp when there is a new release, and we will also update the source code of metal-cpp if there are some critical bugs that need to be fixed.