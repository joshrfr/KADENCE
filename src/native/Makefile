# Native inference kernels, probes, bench and correctness gate.
#   make            build everything (warnings are errors)
#   make test       correctness gate + gate self-test
#   make bench      single command: JSON + table (results/native_bench.json)
#   make CBLAS=1    route the batched dense step through cblas_sgemm (needs cblas.h)
CC      ?= gcc
OPT     ?= -O3
WARN     = -Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes -Wmissing-prototypes \
           -Wconversion -Wno-float-conversion -Werror
CFLAGS   = -std=c11 $(OPT) $(WARN) -Iinclude -D_DEFAULT_SOURCE -ffunction-sections -fdata-sections
LDLIBS   = -lm -Wl,--gc-sections
ifeq ($(CBLAS),1)
CFLAGS  += -DSFN_USE_CBLAS
LDLIBS  += -lcblas
endif

B = build
SRC = src/reserve.c src/ssm.c
OBJ = $(SRC:src/%.c=$(B)/%.o)
PROBES = $(foreach i,0 1 2 3 4 5,$(B)/probe$(i))

all: $(B)/libsfn.a $(B)/libsfn.so $(B)/bench $(PROBES) $(B)/probe3_static

$(B):
	mkdir -p $(B)

$(B)/%.o: src/%.c include/sfn.h | $(B)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

$(B)/libsfn.a: $(OBJ)
	ar rcs $@ $^

$(B)/libsfn.so: $(OBJ)
	$(CC) -shared -o $@ $^ $(LDLIBS)

$(B)/bench: bench/bench.c $(B)/libsfn.a
	$(CC) $(CFLAGS) $< $(B)/libsfn.a $(LDLIBS) -o $@

$(B)/probe%: src/probe.c $(B)/libsfn.a
	$(CC) $(CFLAGS) -DPROBE_IMPL=$* $< $(B)/libsfn.a $(LDLIBS) -o $@

$(B)/probe3_static: src/probe.c $(B)/libsfn.a
	$(CC) $(CFLAGS) -DPROBE_IMPL=3 -static $< $(B)/libsfn.a $(LDLIBS) -o $@

test: all
	python3 tests/gate.py --lib $(B)/libsfn.so
	python3 tests/gate.py --lib $(B)/libsfn.so --selftest
	python3 tests/gate_project.py --lib $(B)/libsfn.so

bench: all
	python3 bench/run_bench.py

clean:
	rm -rf $(B)

.PHONY: all test bench clean
