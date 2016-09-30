BINS := sconv_fprop_K64_N64 sconv_bprop_C64_N64 sconv_update_C128_K128
TARGETS := $(addsuffix .cubin, $(BINS))
TEMPLATES := $(addsuffix _template.cubin, $(BINS))

all: $(BINS)

$(BINS):
	nvcc -arch sm_35 -m 64 $@.cu -cubin -O3 -o $@_template.cubin
	KeplerAs.pl -i $@.sass $@_template.cubin $@.cubin

sconv_fprop: sconv_fprop.cu
	nvcc -arch sm_35 -o $@ $^ -lcuda -lcudart

clean:
	rm $(TARGETS) $(TEMPLATES) sconv_fprop

.PHONY:
	all clean

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true           
