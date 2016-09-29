BINS := sconv_fprop_K64_N64 sconv_bprop_C64_N64 sconv_update_C128_K128
TESTS := sconv_fprop
TARGETS := $(addsuffix .cubin, $(BINS))
TEMPLATES := $(addsuffix _template.cubin, $(BINS))

all: $(BINS) $(TESTS)

$(BINS):
	nvcc -arch sm_35 -m 64 $@.cu -cubin -O3 -o $@_template.cubin
	KeplerAs.pl -i $@.sass $@_template.cubin $@.cubin

$(TESTS):
	nvcc -arch sm_35 $@.cu -o $@ -lcuda -lcudart

clean:
	rm $(TARGETS) $(TEMPLATES) $(TESTS)

.PHONY:
	all clean

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true           
