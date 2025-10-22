ifndef INSTALL_OPT
    INSTALL_OPT = lint,test,dev,docs
endif
ifndef LINT_ARGS
    LINT_ARGS =
endif
ifeq ($(ALL),1)
    LINT_ARGS += --all-files
endif
ifeq ($(VERBOSE),1)
    LINT_ARGS += --verbose
endif

install:
	pip install -e .[$(INSTALL_OPT)]

install_precommit:
	pip install pre-commit && pre-commit install

lint:
	pre-commit run black $(LINT_ARGS) && pre-commit run pylint $(LINT_ARGS) && pre-commit run mypy $(LINT_ARGS)

test:
	pytest

test_slow:
	RUN_SLOW_TESTS=1 pytest

benchmark:
	BENCHMARK=1 pytest . -k benchmark

coverage:
	coverage run -m pytest && coverage html

build_docs:
	./docs/build_docs.sh
