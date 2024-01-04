# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Makefile for common dev-ops tasks.
#
# Helpful `make` options:
# `-n` - Display target command, and don't run it.
# `-B` - Always make all targets.

# Set check options
CHECK_BLACK := true
BLACK_DIRS := .

CHECK_FLAKE8 := true
FLAKE8_DIRS := .

CHECK_MYPY := true
MYPY_DIRS := $(CURDIR)/src/nvidia/srl $(CURDIR)/tests

# Set docs variables
DOCS_OPTS :=
DOCS_SOURCEDIR := docs
DOCS_BUILDDIR := _build/docs/sphinx
DOCS_SOURCES := $(shell find $(DOCS_SOURCEDIR) -type f -not -path "$(DOCS_SOURCEDIR)/_api/*" -not -path "$(DOCS_SOURCEDIR)/api.rst")
DOCS_HTML := $(DOCS_BUILDDIR)/html/index.html

# Set coverage
COVERAGE_SRCDIR := src
COVERAGE_TESTDIR := tests
COVERAGE_BUILDDIR := _build/coverage/py


# Set check target dependencies
CHECK_DEPS =

ifeq ($(CHECK_BLACK),true)
	CHECK_DEPS += check_black
endif

ifeq ($(CHECK_FLAKE8),true)
	CHECK_DEPS += check_flake8
endif

ifeq ($(CHECK_MYPY),true)
	CHECK_DEPS += check_mypy
endif

# Declare phony targets
.PHONY: help, check, check_black, check_flake8, check_mypy, format, test, coverage, package

# Default target: Display a basic help message
help:
	@echo "=================================================="
	@echo "Makefile targets:"
	@echo ""
	@echo "  check       Run code checks."
	@echo "  format      Auto-format code."
	@echo "  test        Run the unit tests."
	@echo "  coverage    Generate a coverage report."
	@echo "  docs        Generate the documentation."
	@echo "  package     Generate a Python package."
	@echo ""
	@echo "Use 'make <target>' to execute a target."
	@echo "=================================================="

# Run the checks
check: $(CHECK_DEPS)

check_black:
	black --diff .

check_flake8:
	flake8 --exit-zero .

check_mypy:
	cd src && \
	for dir in $(MYPY_DIRS); \
	do \
		mypy --config-file=$(CURDIR)/pyproject.toml --namespace-packages --explicit-package-bases $$dir; \
	done

# Run formatter
format:
	isort . && black .

# Run the tests
test:
	pytest .

# Generate the coverage report
coverage:
	pytest --cov-report=term --cov-report=xml:$(COVERAGE_BUILDDIR)/coverage.xml --cov-report=html:$(COVERAGE_BUILDDIR)/html/ --cov=$(COVERAGE_SRCDIR) $(COVERAGE_TESTDIR)

# Generate the documentation
docs: $(DOCS_HTML)

$(DOCS_HTML): $(DOCS_SOURCES)
	sphinx-build -b html $(DOCS_OPTS) $(DOCS_SOURCEDIR) $(DOCS_BUILDDIR)/html

# Generate a Python package
package:
	python3 -m build
