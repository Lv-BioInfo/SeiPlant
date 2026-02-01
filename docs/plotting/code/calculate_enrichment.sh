#!/bin/bash

type=$1
modified_number=${2:-0}  
species_name=$3
skip_files=${4:-""}  # If the fourth parameter is not passed, it defaults to empty.

# Convert skip_files to an array and append the .bed suffix to each filename.
skip_array_with_bed=()
if [ -n "$skip_files" ]; then  # Check whether skip_files is empty
    IFS=',' read -r -a skip_array <<< "$skip_files"
    for skip_file in "${skip_array[@]}"; do
        skip_array_with_bed+=("${skip_file}.bed")
    done
fi

if [ "$modified_number" -ne 0 ]; then
    appendix_name="_modified_$modified_number"
else
    appendix_name=""
fi
type="${type}"
bgsize=374471240
base_dir="/public/workspace/hanquan/liyilin/final_code&data/data/${species_name}"
cluster_bed_dir="/public/workspace/hanquan/liyilin/final_code&data/${species_name}/modified_${modified_number}_prediction_results/cluster_bed${appendix_name}"
marker_dir="${base_dir}/marker"
result_base_dir="/public/workspace/hanquan/liyilin/final_code&data/${species_name}/modified_${modified_number}_prediction_results"
result_dir="${result_base_dir}/${type}${appendix_name}"

mkdir -p "${result_dir}"
> "${result_dir}_list.txt"

for file in $(ls -p "${marker_dir}/${type}" | grep -v /); do
    clean_name=$(echo "${file}" | sed "s/.narrowPeak.bed//; s/.bed//")

    skip=false
    for skip_file in "${skip_array_with_bed[@]}"; do
        if [ "$file" == "$skip_file" ]; then
                skip=true
                break
        fi
    done
        
    # If skipping is required, skip the current file.
    if [ "$skip" = true ]; then
        echo "Skipping file: ${file}"
        continue
    fi    
    echo "${clean_name}" >> "${result_dir}_list.txt"
done

for cluster in $(ls -p "${cluster_bed_dir}" | grep -v /); do
    echo "Processing cluster: ${cluster}"
    total=$(awk -v s=0 '{s+=$3-$2} END {print s}' "${cluster_bed_dir}/${cluster}")
    if [ "${total}" -le 0 ]; then
        echo "Error: total length of ${cluster} is non-positive." >&2
        continue
    fi

    > "${result_dir}/${cluster}"

    for file in $(ls -p "${marker_dir}/${type}" | grep -v /); do
         # Check whether to skip the current file
        skip=false
        for skip_file in "${skip_array_with_bed[@]}"; do
            if [ "$file" == "$skip_file" ]; then
                skip=true
                break
            fi
        done
        
        # If skipping is required, skip the current file.
        if [ "$skip" = true ]; then
            echo "Skipping file: ${file}"
            continue
        fi

        bg=$(awk -v s=0 '{s+=$3-$2} END {print s}' "${marker_dir}/${type}/${file}")
        if [ "${bg}" -le 0 ]; then
            echo "Error: bg length of ${file} is non-positive." >&2
            echo "NA" >> "${result_dir}/${cluster}"
            continue
        fi

        intersect_count=$(bedtools intersect -a "${cluster_bed_dir}/${cluster}" -b "${marker_dir}/${type}/${file}" | wc -l)
        if [ "${intersect_count}" -eq 0 ]; then
            echo "Warning: No intersection for ${file} and ${cluster}." >&2
            echo "NA" >> "${result_dir}/${cluster}"
            continue
        fi

        enrichment_score=$(bedtools intersect -a "${cluster_bed_dir}/${cluster}" -b "${marker_dir}/${type}/${file}" | \
        awk -v FS="\t" -v OFS="\t" -v total=${total} -v bg=${bg} -v bgsize=${bgsize} \
            '{s+=$3-$2} END {if(s==0 || total==0 || bg==0) {print "NA"} else {print log((s/total)/(bg/bgsize))/log(2)}}')
        echo -e "${enrichment_score}" >> "${result_dir}/${cluster}"
    done
done

paste "${result_dir}"/*.bed > "${result_dir}_enrichment.tsv"