/* This file was automatically generated by CasADi 3.6.5.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) Robot_MPC_cost_ext_cost_fun_jac_hess_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s4[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s5[9] = {5, 1, 0, 5, 0, 1, 2, 3, 4};
static const casadi_int casadi_s6[14] = {5, 5, 0, 1, 2, 4, 6, 6, 0, 1, 2, 3, 2, 3};
static const casadi_int casadi_s7[8] = {0, 5, 0, 0, 0, 0, 0, 0};

/* Robot_MPC_cost_ext_cost_fun_jac_hess:(i0[3],i1[2],i2[],i3[13])->(o0,o1[5],o2[5x5,6nz],o3[],o4[0x5]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=1.0000000000000001e-01;
  a1=arg[1]? arg[1][0] : 0;
  a2=casadi_sq(a1);
  a3=arg[1]? arg[1][1] : 0;
  a4=2.6179938779914941e-01;
  a3=(a3/a4);
  a4=casadi_sq(a3);
  a2=(a2+a4);
  a2=(a0*a2);
  a4=5.;
  a5=8.;
  a6=arg[0]? arg[0][0] : 0;
  a5=(a5-a6);
  a7=casadi_sq(a5);
  a8=arg[0]? arg[0][1] : 0;
  a9=casadi_sq(a8);
  a7=(a7+a9);
  a9=256.;
  a7=(a7/a9);
  a4=(a4*a7);
  a2=(a2+a4);
  a4=1.0000000000000000e-03;
  a7=1.;
  a9=10.;
  a10=-1.;
  a10=(a10-a6);
  a6=1.9000000000000001e+00;
  a10=(a10/a6);
  a11=casadi_sq(a10);
  a12=2.0000000000000001e-01;
  a13=(a12-a8);
  a13=(a13/a6);
  a6=casadi_sq(a13);
  a11=(a11+a6);
  a6=1.6000000000000001e+00;
  a11=(a11-a6);
  a11=(a9*a11);
  a11=exp(a11);
  a7=(a7+a11);
  a4=(a4/a7);
  a2=(a2+a4);
  if (res[0]!=0) res[0][0]=a2;
  a1=(a1+a1);
  a1=(a0*a1);
  if (res[1]!=0) res[1][0]=a1;
  a1=3.8197186342054885e+00;
  a3=(a3+a3);
  a0=(a0*a3);
  a1=(a1*a0);
  if (res[1]!=0) res[1][1]=a1;
  a1=5.2631578947368418e-01;
  a0=(a10+a10);
  a3=(a4/a7);
  a2=(a11*a3);
  a2=(a9*a2);
  a6=(a0*a2);
  a6=(a1*a6);
  a14=1.9531250000000000e-02;
  a5=(a5+a5);
  a5=(a14*a5);
  a6=(a6-a5);
  if (res[1]!=0) res[1][2]=a6;
  a6=(a13+a13);
  a5=(a6*a2);
  a5=(a1*a5);
  a8=(a8+a8);
  a14=(a14*a8);
  a5=(a5+a14);
  if (res[1]!=0) res[1][3]=a5;
  a5=0.;
  if (res[1]!=0) res[1][4]=a5;
  if (res[2]!=0) res[2][0]=a12;
  a12=2.9180500888993288e+00;
  if (res[2]!=0) res[2][1]=a12;
  a12=-1.0526315789473684e+00;
  a5=(a12*a2);
  a14=-5.2631578947368418e-01;
  a10=(a10+a10);
  a10=(a14*a10);
  a10=(a9*a10);
  a10=(a11*a10);
  a8=(a3*a10);
  a4=(a4/a7);
  a15=(a4*a10);
  a15=(a15/a7);
  a16=(a3/a7);
  a10=(a16*a10);
  a15=(a15+a10);
  a15=(a11*a15);
  a8=(a8-a15);
  a8=(a9*a8);
  a8=(a0*a8);
  a5=(a5+a8);
  a5=(a1*a5);
  a8=3.9062500000000000e-02;
  a5=(a5+a8);
  if (res[2]!=0) res[2][2]=a5;
  a13=(a13+a13);
  a14=(a14*a13);
  a14=(a9*a14);
  a14=(a11*a14);
  a3=(a3*a14);
  a4=(a4*a14);
  a4=(a4/a7);
  a16=(a16*a14);
  a4=(a4+a16);
  a11=(a11*a4);
  a3=(a3-a11);
  a9=(a9*a3);
  a0=(a0*a9);
  a0=(a1*a0);
  if (res[2]!=0) res[2][3]=a0;
  if (res[2]!=0) res[2][4]=a0;
  a12=(a12*a2);
  a6=(a6*a9);
  a12=(a12+a6);
  a1=(a1*a12);
  a1=(a1+a8);
  if (res[2]!=0) res[2][5]=a1;
  return 0;
}

CASADI_SYMBOL_EXPORT int Robot_MPC_cost_ext_cost_fun_jac_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Robot_MPC_cost_ext_cost_fun_jac_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Robot_MPC_cost_ext_cost_fun_jac_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Robot_MPC_cost_ext_cost_fun_jac_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Robot_MPC_cost_ext_cost_fun_jac_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Robot_MPC_cost_ext_cost_fun_jac_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Robot_MPC_cost_ext_cost_fun_jac_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void Robot_MPC_cost_ext_cost_fun_jac_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Robot_MPC_cost_ext_cost_fun_jac_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Robot_MPC_cost_ext_cost_fun_jac_hess_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real Robot_MPC_cost_ext_cost_fun_jac_hess_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Robot_MPC_cost_ext_cost_fun_jac_hess_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Robot_MPC_cost_ext_cost_fun_jac_hess_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Robot_MPC_cost_ext_cost_fun_jac_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Robot_MPC_cost_ext_cost_fun_jac_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    case 2: return casadi_s6;
    case 3: return casadi_s2;
    case 4: return casadi_s7;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Robot_MPC_cost_ext_cost_fun_jac_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int Robot_MPC_cost_ext_cost_fun_jac_hess_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 5*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
