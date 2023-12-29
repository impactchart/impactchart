X_FEATURES=\
   B25119_003E_2018 \
   frac_B25003A_003E frac_B25003B_003E frac_B25003C_003E \
   frac_B25003D_003E frac_B25003E_003E frac_B25003F_003E \
   frac_B25003G_003E frac_B25003H_003E frac_B25003I_003E

Y_FEATURE=filing_rate

W_FEATURE=B25003_003E

PARAMS=/var/tmp/params.yaml

DATA=evl-34017.csv

PLOT_DIR=/var/tmp/impact

PLOTS = $(X_FEATURES:%=${PLOT_DIR}/%.png)

.PHONY: all plot clean

all: plot

${PARAMS}: ${DATA}
	python -m impactchart.cli.impactchart \
  	optimize \
	-X ${X_FEATURES} \
	-y ${Y_FEATURE} \
	-w ${W_FEATURE} \
	-o $@ \
	$<

plot: ${PLOTS}

${PLOTS}: ${PARAMS} ${DATA}
	python -m impactchart.cli.impactchart \
	--log INFO \
	plot \
	-p ${PARAMS} \
	-X ${X_FEATURES} \
	-y ${Y_FEATURE} \
	-w ${W_FEATURE} \
	-o ${PLOT_DIR} \
	${DATA}

clean:
	rm ${PLOT_DIR}/*
	rm ${PARAMS}
