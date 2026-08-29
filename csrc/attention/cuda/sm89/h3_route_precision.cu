/* SM89 specialization of the verified native H3 route/Anchor producer. */

#define MPA_ROUTE_PRECISION_FUNCTION sm89_h3_route_precision
#define MPA_MATERIALIZE_ROUTE_FUNCTION sm89_h3_materialize_route
#define MPA_ROUTE_DEVICE_OK(properties) \
  ((properties)->major == 8 && (properties)->minor == 9)
#define MPA_ROUTE_DEVICE_LABEL "SM89"
// RTX 4090 tuning is kept independent from the SM120 instantiation even
// though both use the same stable-sort and compact-anchor algorithm.
#define MPA_ROUTE_THREADS 256
// The established SM89 mixed mainloop synthesizes FP16 prefix stages before
// its explicit video route. Preserve that exact ordering for auto/fp16 while
// materializing explicit leading prefix IDs for the INT8 phase.
#define MPA_IMPLICIT_HIGH_PREFIX 1

#include "../sm120/h3_route_precision.cu"
