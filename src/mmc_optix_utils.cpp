#include <cstdlib>
#include <iostream>
#include <time.h>
#include <cstring>
#include <optix_function_table_definition.h>
#include <iomanip>
#include <algorithm>
#include <sutil/vec_math.h>
#include <limits.h>
#ifdef _OPENMP
    #include <omp.h>
#endif

#include "mmc_cuda_query_gpu.h"
#include "mmc_optix_utils.h"
#include "mmc_tictoc.h"
#include "incbin.h"

INCTXT(mmcShaderPtx, mmcShaderPtxSize, "built/mmc_optix_core.ptx");
const int out[4][3] = {{0, 3, 1}, {3, 2, 1}, {0, 2, 3}, {0, 1, 2}};
const int ifaceorder[] = {3, 0, 2, 1};

void optix_run_simulation(mcconfig* cfg, tetmesh* mesh, raytracer* tracer, GPUInfo* gpu,
    void (*progressfun)(float, void*), void* handle) {
    uint tic0 = StartTimer();
    // ==================================================================
    // prepare optix pipeline
    // ==================================================================
    OptixParams optixcfg;

    initOptix();

    createContext(cfg, &optixcfg);
    MMC_FPRINTF(cfg->flog, "optix init complete:  \t%d ms\n", GetTimeMillis() - tic0);
    fflush(cfg->flog);

    std::string ptxcodestr = std::string(mmcShaderPtx);
    createModule(cfg, &optixcfg, ptxcodestr);
    MMC_FPRINTF(cfg->flog, "optix module complete:  \t%d ms\n", GetTimeMillis() - tic0);
    fflush(cfg->flog);

    createRaygenPrograms(&optixcfg);
    createMissPrograms(&optixcfg);
    createHitgroupPrograms(&optixcfg);
    MMC_FPRINTF(cfg->flog, "optix device programs complete:  \t%d ms\n",
        GetTimeMillis() - tic0);
    fflush(cfg->flog);

    extractConnectedRegion(mesh, &optixcfg);
    MMC_FPRINTF(cfg->flog, "extracting connected region complete:  \t%d ms\n", GetTimeMillis() - tic0);
    surfmesh *smesh = (surfmesh*)calloc(optixcfg.regioncount, sizeof(surfmesh));
    prepareSurfMesh(mesh, smesh, &optixcfg);
    MMC_FPRINTF(cfg->flog, "prepare surface mesh complete:  \t%d ms\n", GetTimeMillis() - tic0);
    unsigned int primitiveoffset = 0;
    for (unsigned int i = 0; i < optixcfg.regioncount; ++i) {
        optixcfg.gashandles.push_back(buildAccel(mesh, smesh + i, &optixcfg, primitiveoffset));
        primitiveoffset += smesh[i].norm.size();
    }

    MMC_FPRINTF(cfg->flog, "optix acceleration structure complete:  \t%d ms\n",
        GetTimeMillis() - tic0);
    fflush(cfg->flog);

    createPipeline(&optixcfg);
    MMC_FPRINTF(cfg->flog, "optix pipeline complete:  \t%d ms\n",
        GetTimeMillis() - tic0);
    fflush(cfg->flog);

    buildSBT(mesh, smesh, &optixcfg);
    free(smesh);
    MMC_FPRINTF(cfg->flog, "optix shader binding table complete:  \t%d ms\n",
        GetTimeMillis() - tic0);
    fflush(cfg->flog);

    // ==================================================================
    // prepare launch parameters
    // ==================================================================
    prepLaunchParams(cfg, mesh, gpu, &optixcfg);
    CUDA_ASSERT(cudaDeviceSynchronize());
    MMC_FPRINTF(cfg->flog, "optix launch parameters complete:  \t%d ms\n",
        GetTimeMillis() - tic0);
    fflush(cfg->flog);

    // ==================================================================
    // Launch simulation
    // ==================================================================
    MMC_FPRINTF(cfg->flog, "lauching OptiX for time window [%.1fns %.1fns] ...\n",
        cfg->tstart * 1e9, cfg->tend * 1e9);
    fflush(cfg->flog);
    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                        optixcfg.pipeline,
                        optixcfg.stream,
                        /*! parameters and SBT */
                        optixcfg.launchParamsBuffer.d_pointer(),
                        optixcfg.launchParamsBuffer.sizeInBytes,
                        &optixcfg.sbt,
                        /*! dimensions of the launch: */
                        optixcfg.launchWidth, 1, 1));
    CUDA_ASSERT(cudaDeviceSynchronize());
    MMC_FPRINTF(cfg->flog, "kernel complete:  \t%d ms\nretrieving flux ... \t",
        GetTimeMillis() - tic0);
    fflush(cfg->flog);

    // ==================================================================
    // Save output
    // ==================================================================
    optixcfg.outputBuffer.download(optixcfg.outputHostBuffer, optixcfg.outputBufferSize);
    MMC_FPRINTF(cfg->flog, "transfer complete:        %d ms\n", GetTimeMillis() - tic0);
    fflush(cfg->flog);
    for (size_t i = 0; i < optixcfg.launchParams.crop0.w; i++) {
        // combine two outputs into one
        #pragma omp atomic
        mesh->weight[i] += optixcfg.outputHostBuffer[i] +
            optixcfg.outputHostBuffer[i + optixcfg.launchParams.crop0.w];
    }

    // ==================================================================
    // normalize output
    // ==================================================================
    if (cfg->isnormalized) {
        MMC_FPRINTF(cfg->flog, "normalizing raw data ...\t");
        fflush(cfg->flog);

        // not used if cfg->method == rtBLBadouelGrid
        cfg->energyabs = 0.0f;

        // for now assume initial weight of each photon is 1.0
        cfg->energytot = cfg->nphoton;
        mesh_normalize(mesh, cfg, cfg->energyabs, cfg->energytot, 0);
        MMC_FPRINTF(cfg->flog, "normalization complete:    %d ms\n",
            GetTimeMillis() - tic0);
        fflush(cfg->flog);
    }

    #pragma omp master
    {
        if (cfg->issave2pt && cfg->parentid == mpStandalone) {
            MMC_FPRINTF(cfg->flog, "saving data to file ...\t");
            mesh_saveweight(mesh, cfg, 0);
            MMC_FPRINTF(cfg->flog, "saving data complete : %d ms\n\n",
                        GetTimeMillis() - tic0);
            fflush(cfg->flog);
        }
    }

    // ==================================================================
    // Free memory
    // ==================================================================
    clearOptixParams(&optixcfg);
}

/**<
 * @brief Label the tetrahedron elements in each simply connected region
 */
void extractConnectedRegion(tetmesh *tmesh, OptixParams *optixcfg) {
    // -1 indicates that the element is not labeled yet
    optixcfg->connectedregion.resize(tmesh->ne, -1);
    unsigned int regionid = 1; // start from 1 because 0 will be used by the region outside
    for (int i = 0; i < tmesh->ne; ++i) {
        if (optixcfg->connectedregion[i] == -1) {
            // if the ith element is not labeled yet, we find a new region
            findSimplyConnectedRegion(tmesh, optixcfg->connectedregion, regionid++, i);
        }
    }
    optixcfg->regioncount = regionid;
}

/**<
 * @brief Use DFS to find the simply connected region for a tetrahedron element
 */
void findSimplyConnectedRegion(tetmesh *tmesh, std::vector<int> &connectedregion,
    int regionid, int elemid) {
    // label the element
    connectedregion[elemid] = regionid;

    // check the neighbouring elements
    unsigned int currmedid = tmesh->type[elemid];
    for (int i = 0; i < tmesh->elemlen; ++i) {
        int nextelemid = tmesh->facenb[(elemid * tmesh->elemlen) + i] - 1;
        if (nextelemid >= 0) {
            unsigned int nextmedid = tmesh->type[nextelemid];
            if (nextmedid == currmedid && connectedregion[nextelemid] != regionid) {
                findSimplyConnectedRegion(tmesh, connectedregion, regionid, nextelemid);
            }
        }
    }
}

/**
 * @brief extract surface mesh for each medium
 */
void prepareSurfMesh(tetmesh *tmesh, surfmesh *smesh, OptixParams *optixcfg) {
    int *fnb = (int*)calloc(tmesh->ne * tmesh->elemlen, sizeof(int));
    memcpy(fnb, tmesh->facenb, (tmesh->ne * tmesh->elemlen) * sizeof(int));

    float3 v0, v1, v2, vec01, vec02, vnorm;
    for (int i = 0; i < tmesh->ne; ++i) {
        // iterate over each tetrahedra
        unsigned int currmedid = tmesh->type[i];
        unsigned int currregionid = optixcfg->connectedregion[i];
        for(int j = 0; j < tmesh->elemlen; ++j){
            // iterate over each triangle
            int nexteid = fnb[(i * tmesh->elemlen) + j];
            if (nexteid == INT_MIN) continue;
            unsigned int nextmedid = ((nexteid < 0) ? 0 : tmesh->type[nexteid - 1]);
            unsigned int nextregionid = ((nexteid < 0) ? 0 : optixcfg->connectedregion[nexteid - 1]);
            if(currregionid != nextregionid) {
                // face nodes
                unsigned int n0 = tmesh->elem[(i * tmesh->elemlen) + out[ifaceorder[j]][0]] - 1;
                unsigned int n1 = tmesh->elem[(i * tmesh->elemlen) + out[ifaceorder[j]][1]] - 1;
                unsigned int n2 = tmesh->elem[(i * tmesh->elemlen) + out[ifaceorder[j]][2]] - 1;

                // face vertex indices
                smesh[currregionid].face.push_back(make_uint3(n0, n1, n2));
                smesh[nextregionid].face.push_back(make_uint3(n0, n2, n1));

                // outward-pointing face norm
                v0 = *(float3*)&tmesh->fnode[n0];
                v1 = *(float3*)&tmesh->fnode[n1];
                v2 = *(float3*)&tmesh->fnode[n2];
                vec_diff((MMCfloat3*)&v0, (MMCfloat3*)&v1, (MMCfloat3*)&vec01);
                vec_diff((MMCfloat3*)&v0, (MMCfloat3*)&v2, (MMCfloat3*)&vec02);
                vec_cross((MMCfloat3*)&vec01, (MMCfloat3*)&vec02, (MMCfloat3*)&vnorm);
                float mag = 1.0f / sqrtf(vec_dot((MMCfloat3*)&vnorm, (MMCfloat3*)&vnorm));
                vec_mult((MMCfloat3*)&vnorm, mag, (MMCfloat3*)&vnorm);
                smesh[currregionid].norm.push_back(vnorm);
                smesh[nextregionid].norm.push_back(-vnorm);

                // neighbour medium types
                smesh[currregionid].nbtype.push_back(nextmedid);
                smesh[nextregionid].nbtype.push_back(currmedid);

                // neighbour region id
                smesh[currregionid].nbregion.push_back(nextregionid);
                smesh[nextregionid].nbregion.push_back(currregionid);

                fnb[(i * tmesh->elemlen) + j] = INT_MIN;
                if(nexteid > 0){
                    for(int k = 0; k < tmesh->elemlen; ++k){
                        if(fnb[((nexteid - 1) * tmesh->elemlen) + k] == i + 1) {
                            fnb[((nexteid - 1) * tmesh->elemlen) + k] = INT_MIN;
                            break;
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief prepare launch parameters
 */
void prepLaunchParams(mcconfig* cfg, tetmesh* mesh, GPUInfo* gpu, OptixParams *optixcfg) {
    if (cfg->method != rtBLBadouelGrid) {
        mcx_error(-1, "Optix MMC only supports dual grid mode", __FILE__, __LINE__);
    }

    int timeSteps = (int)((cfg->tend - cfg->tstart) / cfg->tstep + 0.5);
    if (timeSteps < 1) {
        mcx_error(-1, "There must be at least one time step.", __FILE__, __LINE__);
    }

    // set up optical properties
    if (mesh->prop + 1 > MAX_PROP_OPTIX) {
        mcx_error(-1, "Medium type count exceeds limit.", __FILE__, __LINE__);
    }
    for (int i = 0; i <= mesh->prop; ++i) {
        optixcfg->launchParams.medium[i].mua = mesh->med[i].mua;
        optixcfg->launchParams.medium[i].mus = mesh->med[i].mus;
        optixcfg->launchParams.medium[i].g = mesh->med[i].g;
        optixcfg->launchParams.medium[i].n = mesh->med[i].n;
    }

    // source setup
    optixcfg->launchParams.srcpos = make_float3(cfg->srcpos.x,
                                                cfg->srcpos.y,
                                                cfg->srcpos.z);
    optixcfg->launchParams.srcdir = make_float3(cfg->srcdir.x,
                                                cfg->srcdir.y,
                                                cfg->srcdir.z);

    // parameters of dual grid
    optixcfg->launchParams.nmin = make_float3(mesh->nmin.x,
                                              mesh->nmin.y,
                                              mesh->nmin.z);
    optixcfg->launchParams.nmax = make_float3(mesh->nmax.x - mesh->nmin.x,
                                              mesh->nmax.y - mesh->nmin.y,
                                              mesh->nmax.z - mesh->nmin.z);
    optixcfg->launchParams.crop0 = make_uint4(cfg->crop0.x,
                                               cfg->crop0.y,
                                               cfg->crop0.z,
                                               cfg->crop0.z * timeSteps);
    optixcfg->launchParams.dstep = 1.0f / cfg->unitinmm;

    // time-gate settings
    optixcfg->launchParams.tstart = cfg->tstart;
    optixcfg->launchParams.tend = cfg->tend;
    optixcfg->launchParams.Rtstep = 1.0f / cfg->tstep;
    optixcfg->launchParams.maxgate = cfg->maxgate;

    // init medium ID using element based
    optixcfg->launchParams.mediumid0 = mesh->type[cfg->e0-1];

    // init gashandle using initial medium ID
    optixcfg->launchParams.gashandle0 =
        optixcfg->gashandles[optixcfg->connectedregion[cfg->e0-1]];

    // simulation flags
    optixcfg->launchParams.isreflect = cfg->isreflect;

    // output type
    optixcfg->launchParams.outputtype = static_cast<int>(cfg->outputtype);

    // number of photons for each thread
    int totalthread = cfg->nthread;

    int gpuid, threadid = 0;
#ifdef _OPENMP
    threadid = omp_get_thread_num();
#endif
    gpuid = cfg->deviceid[threadid] - 1;

    if (cfg->autopilot)
        totalthread = gpu[gpuid].autothread;

    optixcfg->launchWidth = totalthread;
    optixcfg->launchParams.threadphoton = cfg->nphoton / optixcfg->launchWidth;
    optixcfg->launchParams.oddphoton =
        cfg->nphoton - optixcfg->launchParams.threadphoton * totalthread;

    // output buffer (single precision)
    optixcfg->outputBufferSize = (optixcfg->launchParams.crop0.w << 1);
    optixcfg->outputHostBuffer = (float*)calloc(optixcfg->outputBufferSize, sizeof(float));
    optixcfg->outputBuffer.alloc_and_upload(optixcfg->outputHostBuffer,
        optixcfg->outputBufferSize);
    optixcfg->launchParams.outputbuffer = optixcfg->outputBuffer.d_pointer();

    // photon seed buffer
    if (cfg->seed > 0) {
        srand(cfg->seed);
    } else {
        srand(time(0));
    }
    uint4 *hseed = (uint4 *)calloc(totalthread, sizeof(uint4));
    for (int i = 0; i < totalthread; ++i) {
        hseed[i] = make_uint4(rand(), rand(), rand(), rand());
    }
    optixcfg->seedBuffer.alloc_and_upload(hseed, totalthread);
    optixcfg->launchParams.seedbuffer = optixcfg->seedBuffer.d_pointer();
    if (hseed) free(hseed);

    // upload launch parameters to device
    optixcfg->launchParamsBuffer.alloc_and_upload(&optixcfg->launchParams, 1);
}

/**************************************************************************
 * helper functions for Optix pipeline creation
******************************************************************************/

/**
 * @brief initialize optix
 */
void initOptix() {
    cudaFree(0);
    OPTIX_CHECK(optixInit());
}

/**
 * @brief creates and configures a optix device context
 */
void createContext(mcconfig* cfg, OptixParams* optixcfg) {
    int gpuid, threadid = 0;

#ifdef _OPENMP
    threadid = omp_get_thread_num();
#endif

    gpuid = cfg->deviceid[threadid] - 1;
    if (gpuid < 0) {
        mcx_error(-1, "GPU ID must be non-zero", __FILE__, __LINE__);
    }

    CUDA_ASSERT(cudaSetDevice(gpuid));
    CUDA_ASSERT(cudaStreamCreate(&optixcfg->stream));

    cudaGetDeviceProperties(&optixcfg->deviceProps, gpuid);
    std::cout << "Running on device: " << optixcfg->deviceProps.name << std::endl;

    CUresult cuRes = cuCtxGetCurrent(&optixcfg->cudaContext);
    if(cuRes != CUDA_SUCCESS)
        fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = [](unsigned int level, const char* tag, const char* message,
                                      void*) {
                                      std::cerr << "[" << std::setw( 2 ) << level
                                          << "][" << std::setw( 12 ) << tag << "]: "
                                          << message << "\n";
                                  };
#ifndef NDEBUG
    options.logCallbackLevel = 4;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
    options.logCallbackLevel = 0;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
#endif

    OPTIX_CHECK(optixDeviceContextCreate(optixcfg->cudaContext, &options,
        &optixcfg->optixContext));
}

/**
 * @brief creates the module that contains all programs
 */
void createModule(mcconfig* cfg, OptixParams* optixcfg, std::string ptxcode) {
    // moduleCompileOptions
    optixcfg->moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    optixcfg->moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    optixcfg->moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
#else
    optixcfg->moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    optixcfg->moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#endif

    // pipelineCompileOptions
    optixcfg->pipelineCompileOptions = {};
    optixcfg->pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    optixcfg->pipelineCompileOptions.usesMotionBlur     = false;
    optixcfg->pipelineCompileOptions.numPayloadValues   = 16;
    optixcfg->pipelineCompileOptions.numAttributeValues = 2;  // for triangle
#ifndef NDEBUG
    optixcfg->pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG |
        OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    optixcfg->pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    optixcfg->pipelineCompileOptions.pipelineLaunchParamsVariableName = "gcfg";

    // pipelineLinkOptions
    optixcfg->pipelineLinkOptions.maxTraceDepth = 1;

    char log[2048];
    size_t logsize = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(optixcfg->optixContext,
                                         &optixcfg->moduleCompileOptions,
                                         &optixcfg->pipelineCompileOptions,
                                         ptxcode.c_str(),
                                         ptxcode.size(),
                                         log,
                                         &logsize,
                                         &optixcfg->module
                                         ));
    if (logsize > 1) std::cout << log << std::endl;
}

/**
 * @brief set up ray generation programs
 */
void createRaygenPrograms(OptixParams* optixcfg) {
    optixcfg->raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = optixcfg->module;
    pgDesc.raygen.entryFunctionName = "__raygen__rg";

    char log[2048];
    size_t logsize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixcfg->optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,
                                        &logsize,
                                        &optixcfg->raygenPGs[0]
                                        ));
    if (logsize > 1) std::cout << log << std::endl;
}

/**
 * @brief set up miss programs
 */
void createMissPrograms(OptixParams* optixcfg) {
    optixcfg->missPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.raygen.module            = optixcfg->module;
    pgDesc.raygen.entryFunctionName = "__miss__ms";

    char log[2048];
    size_t logsize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixcfg->optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,
                                        &logsize,
                                        &optixcfg->missPGs[0]
                                        ));
    if (logsize > 1) std::cout << log << std::endl;
}

/**
 * @brief set up hitgroup programs
 */
void createHitgroupPrograms(OptixParams* optixcfg) {
    optixcfg->hitgroupPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = optixcfg->module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

    char log[2048];
    size_t logsize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixcfg->optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,
                                        &logsize,
                                        &optixcfg->hitgroupPGs[0]
                                        ));
    if (logsize > 1) std::cout << log << std::endl;
}

/**
 * @brief set up acceleration structures
 */
OptixTraversableHandle buildAccel(tetmesh *tmesh, surfmesh* smesh, OptixParams* optixcfg,
    unsigned int primitiveoffset) {
    OptixTraversableHandle asHandle {0};
    if (smesh->face.empty()) return asHandle;
    // ==================================================================
    // upload the model to the device
    // note: mesh->fnode needs to be float3
    // mesh->face needs to be uint3 (zero-indexed)
    // ==================================================================
    optixcfg->vertexBuffer.alloc_and_upload(tmesh->fnode, tmesh->nn);
    optixcfg->indexBuffer.alloc_and_upload(smesh->face);

    // ==================================================================
    // triangle inputs
    // ==================================================================
    OptixBuildInput triangleInput = {};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    CUdeviceptr d_vertices = optixcfg->vertexBuffer.d_pointer();
    CUdeviceptr d_indices  = optixcfg->indexBuffer.d_pointer();

    triangleInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangleInput.triangleArray.numVertices         = tmesh->nn;
    triangleInput.triangleArray.vertexBuffers       = &d_vertices;

    triangleInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes  = sizeof(uint3);
    triangleInput.triangleArray.numIndexTriplets    = smesh->face.size();
    triangleInput.triangleArray.indexBuffer         = d_indices;

    uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

    // in this example we have one SBT entry, and no per-primitive
    // materials:
    triangleInput.triangleArray.flags               = triangleInputFlags;
    triangleInput.triangleArray.numSbtRecords               = 1;
    triangleInput.triangleArray.sbtIndexOffsetBuffer        = 0;
    triangleInput.triangleArray.sbtIndexOffsetSizeInBytes   = 0;
    triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;
    triangleInput.triangleArray.primitiveIndexOffset = primitiveoffset;

    // ==================================================================
    // BLAS setup
    // ==================================================================
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE
      | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
      ;
    accelOptions.motionOptions.numKeys  = 1;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optixcfg->optixContext,
                 &accelOptions,
                 &triangleInput,
                 1,  // num_build_inputs
                 &blasBufferSizes
                 ));

    // ==================================================================
    // prepare compaction
    // ==================================================================
    osc::CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================
    osc::CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    osc::CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(optixcfg->optixContext,
                                optixcfg->stream,
                                &accelOptions,
                                &triangleInput,
                                1,
                                tempBuffer.d_pointer(),
                                tempBuffer.sizeInBytes,
                                outputBuffer.d_pointer(),
                                outputBuffer.sizeInBytes,
                                &asHandle,
                                &emitDesc,1
                                ));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize,1);

    optixcfg->asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixcfg->optixContext,
                                  optixcfg->stream,
                                  asHandle,
                                  optixcfg->asBuffer.d_pointer(),
                                  optixcfg->asBuffer.sizeInBytes,
                                  &asHandle));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;
}

/**
 * @brief assemble the pipeline of all programs
 */
void createPipeline(OptixParams* optixcfg) {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : optixcfg->raygenPGs)
      programGroups.push_back(pg);
    for (auto pg : optixcfg->missPGs)
      programGroups.push_back(pg);
    for (auto pg : optixcfg->hitgroupPGs)
      programGroups.push_back(pg);

    char log[2048];
    size_t logsize = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(optixcfg->optixContext,
                                    &optixcfg->pipelineCompileOptions,
                                    &optixcfg->pipelineLinkOptions,
                                    programGroups.data(),
                                    (int)programGroups.size(),
                                    log,
                                    &logsize,
                                    &optixcfg->pipeline
                                    ));
    if (logsize > 1) std::cout << log << std::endl;
}

/**
 * @ set up the shader binding table
 */
void buildSBT(tetmesh* tmesh, surfmesh* smesh, OptixParams* optixcfg) {
    // ==================================================================
    // build raygen records
    // ==================================================================
    std::vector<RaygenRecord> raygenRecords;
    for (size_t i = 0;i < optixcfg->raygenPGs.size();i++) {
      RaygenRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(optixcfg->raygenPGs[i],&rec));
      rec.data = nullptr;
      raygenRecords.push_back(rec);
    }
    optixcfg->raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    optixcfg->sbt.raygenRecord = optixcfg->raygenRecordsBuffer.d_pointer();

    // ==================================================================
    // build miss records
    // ==================================================================
    std::vector<MissRecord> missRecords;
    for (size_t i = 0;i < optixcfg->missPGs.size();i++) {
      MissRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(optixcfg->missPGs[i],&rec));
      rec.data = nullptr; /* for now ... */
      missRecords.push_back(rec);
    }
    optixcfg->missRecordsBuffer.alloc_and_upload(missRecords);
    optixcfg->sbt.missRecordBase          = optixcfg->missRecordsBuffer.d_pointer();
    optixcfg->sbt.missRecordStrideInBytes = sizeof(MissRecord);
    optixcfg->sbt.missRecordCount         = (int)missRecords.size();

    // ==================================================================
    // build hitgroup records
    // ==================================================================
    std::vector<HitgroupRecord> hitgroupRecords;
    HitgroupRecord rec;
    // all meshes use the same code, so all same hit group
    OPTIX_CHECK(optixSbtRecordPackHeader(optixcfg->hitgroupPGs[0],&rec));

    // combine face normal + front + back into a float4 array
    std::vector<float4> fnorm;
    std::vector<OptixTraversableHandle> nbgashandle;
    for (unsigned int i = 0; i < optixcfg->regioncount; ++i) {
        for (size_t j = 0; j < smesh[i].norm.size(); ++j) {
            fnorm.push_back(make_float4(smesh[i].norm[j].x, smesh[i].norm[j].y,
                smesh[i].norm[j].z, *(float*)&smesh[i].nbtype[j]));
            nbgashandle.push_back(optixcfg->gashandles[smesh[i].nbregion[j]]);
        }
    }
    optixcfg->fnormBuffer.alloc_and_upload(fnorm);
    optixcfg->nbgashandleBuffer.alloc_and_upload(nbgashandle);
    rec.data.fnorm = (float4*)optixcfg->fnormBuffer.d_pointer();
    rec.data.nbgashandle = (OptixTraversableHandle*)optixcfg->nbgashandleBuffer.d_pointer();
    hitgroupRecords.push_back(rec);

    optixcfg->hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    optixcfg->sbt.hitgroupRecordBase          = optixcfg->hitgroupRecordsBuffer.d_pointer();
    optixcfg->sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    optixcfg->sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
}

/**
 * @ Free allocated memory
 */
void clearOptixParams(OptixParams* optixcfg) {
    optixcfg->raygenRecordsBuffer.free();
    optixcfg->missRecordsBuffer.free();
    optixcfg->hitgroupRecordsBuffer.free();
    optixcfg->launchParamsBuffer.free();
    optixcfg->vertexBuffer.free();
    optixcfg->indexBuffer.free();
    optixcfg->fnormBuffer.free();
    optixcfg->nbgashandleBuffer.free();
    optixcfg->asBuffer.free();
    optixcfg->seedBuffer.free();
    optixcfg->outputBuffer.free();
    free(optixcfg->outputHostBuffer);
}