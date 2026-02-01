import numpy as np
import os
def main(base_path='../ath/prediction_results/20250214_184449/',modified_number = 0,fa_file_path="/public/workspace/hanquan/liyilin/merged_tag/training_fa/mergedtag_ath_1024_128.fa" ):

    if modified_number != 0:
        appendix_name = f'_modified_{modified_number}'
    else:
        appendix_name = ''

    leiden_file_name = 'new_data_predictions'+ appendix_name + '.leiden30.npy'
    leiden_path = os.path.join(base_path,leiden_file_name)

    def parse_fa_file(fa_file_path):
        """
        Parse FA files and return a list containing the chromosome names, start, and end positions for all sequences.。
        """
        sequences = []
        with open(fa_file_path, 'r') as fa_file:
            lines = fa_file.readlines()
            for i in range(0, len(lines), 2):  # Every two lines form a sequence.
                header = lines[i].strip()  # Sequence header
                # sequence = lines[i+1].strip()  # Sequence content (ATCG), but we only need header information here, not the actual sequence

                # Parse chromosome information from the header, extracting chromosome name, start, and end
                header_parts = header.split('::')  # Separated by '::'
                chrom_info = header_parts[1]  # Extract the second part: Chromosome name:start-end
                chrom_name, start_end = chrom_info.split(':')  # Extract chromosome names and start-end positions
                chrom_name = chrom_name.split("_")[0]
                start, end = map(int, start_end.split('-'))  # Parse start and end as integers

                # Store as a (chromosome name, start, end) tuple
                sequences.append((chrom_name, start, end))

        return sequences


    def create_cluster_bed(npy_file_path, fa_file_path, output_dir):
        """
        Generate the cluster.bed file based on the Leiden class in the npy file.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 1. Load the npy file to obtain the index of the Leiden category.
        leiden_inds = np.load(npy_file_path)

        # 2. Parse the FA file to obtain the chromosome names, start positions, and end positions for all sequences.
        sequences = parse_fa_file(fa_file_path)

        # 3. Verify that the number of entries in leiden_inds matches the sequence count.
        if len(leiden_inds) != len(sequences):
            raise ValueError(f"The length of leiden_inds ({len(leiden_inds)}) does not match the number of sequences in the FA file ({len(sequences)})!")

        # 4. Retrieve all unique Leiden categories
        unique_labels = np.unique(leiden_inds)  
        cluster_list = []
        cluster_list_path = os.path.join(output_dir, "cluster_list"+ appendix_name + ".txt") 
        #  5. Create a bed file for each Leiden category.
        for label in unique_labels:
            cluster_filename = f"{output_dir}/cluster{label}.bed"
            with open(cluster_filename, 'w') as bed_file:
                # Find the indices of all sequences belonging to the current category.
                label_inds = np.where(leiden_inds == label)[0]

                # Based on the index of the current category, extract the corresponding sequence information from sequences.
                for idx in label_inds:
                    chrom_name, start, end = sequences[idx]
                    # Write to the BED file
                    bed_file.write(f"{chrom_name}\t{start}\t{end}\tCluster{label}\n")

            cluster_name = f"cluster{label}"+' '+f"{label}"  # Remove the suffix
            cluster_list.append(cluster_name)

            print(f"Cluster {label} saved to {cluster_filename}")
        

        with open(cluster_list_path, "w") as file:
            i=0
            # Iterate through each element in the list
            for item in cluster_list:
                # Write each element to the file, with each element occupying one line.
                file.write(item + "\n")
                i+=1
            print(f'{i}clusters in total!')
        
    prediction_results_path = os.path.join(base_path, 'new_data_predictions'+appendix_name+'.npy')
    prediction_results = np.load(prediction_results_path)
    print(len(prediction_results))


    npy_file_path = leiden_path  # Path to the .npy file for the Leiden-style labels
    # Create a new cluster_bed file
    output_dir = os.path.join(base_path,"cluster_bed"+appendix_name) #  Directory of the output cluster.bed files

    create_cluster_bed(npy_file_path, fa_file_path, output_dir)
    print('Cluster files have been created.')
    
if __name__ == "__main__":
    # Configure command-line argument parsing
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--base_path", type=str, default='../ath/prediction_results/20250214_184449/', help="Base path for the prediction results")
    parser.add_argument("--modified_number", type=int, default=0, help="Modified number for the file name")
    parser.add_argument("--fa_file_path", type=str, default="/public/workspace/hanquan/liyilin/merged_tag/training_fa/mergedtag_ath_1024_128.fa", help="Path to the FASTA file")
    
    # Parsing Command-Line Arguments
    args = parser.parse_args()
    
    # Call the main function
    main(base_path=args.base_path, modified_number=args.modified_number, fa_file_path=args.fa_file_path)

