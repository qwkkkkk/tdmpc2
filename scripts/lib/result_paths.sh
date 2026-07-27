#!/bin/bash

# Canonical experiment result paths. Existing flat layouts remain readable so
# completed checkpoints can be reused without moving large artifact trees.

tdmpc2_clean_dir() {
    local root=$1 domain=$2 task=$3 run_name=$4
    printf '%s/logs/%s/%s/clean/tdmpc2/%s\n' \
        "${root}" "${domain}" "${task}" "${run_name}"
}

tdmpc2_legacy_clean_dir() {
    local root=$1 domain=$2 run_name=$3
    printf '%s/logs/%s/clean/%s\n' "${root}" "${domain}" "${run_name}"
}

tdmpc2_backdoor_dir() {
    local root=$1 domain=$2 task=$3 method=$4 run_name=$5
    printf '%s/logs/%s/%s/backdoor/%s/%s\n' \
        "${root}" "${domain}" "${task}" "${method}" "${run_name}"
}

tdmpc2_legacy_backdoor_dir() {
    local root=$1 domain=$2 run_name=$3
    printf '%s/logs/%s/backdoor/%s\n' "${root}" "${domain}" "${run_name}"
}

tdmpc2_prefer_existing_dir() {
    local canonical=$1 legacy=$2 marker=$3
    if [[ -f "${canonical}/${marker}" ]]; then
        printf '%s\n' "${canonical}"
    elif [[ -f "${legacy}/${marker}" ]]; then
        printf '%s\n' "${legacy}"
    elif [[ -d "${canonical}" ]]; then
        printf '%s\n' "${canonical}"
    elif [[ -d "${legacy}" ]]; then
        printf '%s\n' "${legacy}"
    else
        printf '%s\n' "${canonical}"
    fi
}
