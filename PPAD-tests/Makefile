run:
	futhark bench --backend=cuda test-scan-mm2.fut --pass-option=--default-group-size=512 --json=mm2.json
	futhark bench --backend=cuda test-scan-mm3.fut --pass-option=--default-group-size=512 --json=mm3.json
	futhark bench --backend=cuda test-scan-mm5.fut --pass-option=--default-group-size=512 --json=mm5.json
	futhark bench --backend=cuda scan-lfc-opw.fut --pass-option=--default-group-size=512 --json=lfc.json
	futhark bench --backend=cuda scan-fc-opw.fut --pass-option=--default-group-size=512 --json=fc.json
	futhark bench --backend=cuda scan-basic-ops.fut --pass-option=--default-group-size=512 --json=basic.json
