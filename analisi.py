import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import csv
import numpy as np

FILE = "./test-metrics.csv"

BEST_TH     = "20"
BEST_BS 	= "128"
MAX_THREADS = 40
THREADS = range(1, MAX_THREADS + 1)
BLOCKS  = [128,256,384,512,640,768,896,1024]


def get_num_threads(th):
	if th == '4':
		return 0
	elif th == '8':
		return 1
	elif th == '16':
		return 2
	elif th == '32':
		return 3
	else:
		return -1	

def get_block_size(bs):
	if bs == '128':
		return 0
	elif bs == '256':
		return 1
	elif bs == '512':
		return 2
	elif bs == '1024':
		return 3
	else:
		return -1


def get_serial_calc_time(matrix_name):
	dictonary = csv.DictReader(open(FILE, newline=''))
	for row in dictonary:
		if( row["CalculationMode"] == "-serial" and row["Matrix"] == matrix_name ):
			return row["CalculationTime(ms)"]

def get_cuda_csr_time(matrix_name):
	cuda_csr_time = 0
	dictonary = csv.DictReader(open(FILE, newline=''))
	for row in dictonary:
		if( "-cudaCSR" in row["CalculationMode"] and row["Matrix"] == matrix_name and row['Threads'] == BEST_BS):
				cuda_csr_time = row["CalculationTime(ms)"]
	return cuda_csr_time

def get_cuda_ellpack_time(matrix_name):
	cuda_ellpack_time = 0
	dictonary = csv.DictReader(open(FILE, newline=''))
	for row in dictonary:
		if( "-cudaELLPACK" in row["CalculationMode"] and row["Matrix"] == matrix_name and row['Threads'] == BEST_BS):
				cuda_ellpack_time = row["CalculationTime(ms)"]
	return  cuda_ellpack_time


def print_table(keys1, keys2, val1, val2, mode):
	
	print("\\hline")
	print("Matrice & %s \\\\ [1ex] "% (mode))
	print("\\hline")
	for key in keys1:
		ind1 = keys1.index(key)
		if key in keys2:
			ind2 = keys2.index(key)
			print("%s & %.2f & %.2f \\\\ "% (key, val1[ind1], val2[ind2]))
			print("\\hline")
		else:
			print("%s & %.2f &  \\\\ "% (key, val1[ind1]))
			print("\\hline")



def cpu_istogramma_speedup():

	dictonary = csv.DictReader(open(FILE, newline=''))

	cpu_csr = {}
	cpu_ellpack = {}

	for row in dictonary:
		mode = row['CalculationMode']
		threads = row['Threads']
		if( "-ompCSR" in mode ):
			#csr
			name = row["Matrix"]
			serial_time = get_serial_calc_time(name)
			speedup = float(serial_time) / float(row["CalculationTime(ms)"])
			matrix = cpu_csr.get(name)
			if ( index := get_num_threads(threads)) == -1:
				continue
			if matrix:
				matrix[index] = speedup
			else:
				matrix = [0,0,0,0]
				matrix[index] = speedup
			cpu_csr[name] = matrix

		elif( "-ompELLPACK" in mode ):
			#ellpack
			name = row["Matrix"]
			serial_time = get_serial_calc_time(name)
			speedup = float(serial_time) / float(row["CalculationTime(ms)"])
			matrix = cpu_ellpack.get(name)
			if ( index := get_num_threads(threads)) == -1:
				continue
			if matrix:
				matrix[index] = speedup
			else:
				matrix = [0,0,0,0]
				matrix[index] = speedup
			cpu_ellpack[name] = matrix
	
	keys = [key[1:-4] for key in cpu_csr.keys()]
	values = [value for value in cpu_csr.values()]
	fig, ax = plt.subplots()
	th4 = ax.bar(np.arange(len(keys)) - 0.3, [value[0] for value in values],
	       width=0.2, color='b', align='center')
	th8 = ax.bar(np.arange(len(keys)) - 0.1,
	       [value[1] for value in values],
	       width=0.2, color='g', align='center')
	th16 = ax.bar(np.arange(len(keys)) + 0.1,
	       [value[2] for value in values],
	       width=0.2, color='r', align='center')
	th32 = ax.bar(np.arange(len(keys)) + 0.3,
	       [value[3] for value in values],
	       width=0.2, color='orange', align='center')
	ax.legend([th4, th8, th16, th32], ['4 threads', '8 threads', '16 threads', '32 threads'])
	ax.set_xticklabels(keys, rotation='vertical', fontsize=8)
	ax.set_xticks(np.arange(len(keys)), rotation='vertical', fontsize=8)
	
	plt.xlabel("Matrice", fontsize=11)
	plt.ylabel("Speedup", fontsize=11)
	plt.title("CPU CSR performance", fontsize=13)

	print(keys)

	plt.show()

	ell_keys = [key[1:-4] for key in cpu_ellpack.keys()]
	ell_values = [value for value in cpu_ellpack.values()]
	fig, ax = plt.subplots()
	th4 = ax.bar(np.arange(len(ell_keys)) - 0.3, [value[0] for value in ell_values],
	       width=0.2, color='b', align='center')
	th8 = ax.bar(np.arange(len(ell_keys)) - 0.1,
	       [value[1] for value in ell_values],
	       width=0.2, color='g', align='center')
	th16 = ax.bar(np.arange(len(ell_keys)) + 0.1,
	       [value[2] for value in ell_values],
	       width=0.2, color='r', align='center')
	th32 = ax.bar(np.arange(len(ell_keys)) + 0.3,
	       [value[3] for value in ell_values],
	       width=0.2, color='orange', align='center')
	ax.legend([th4, th8, th16, th32], ['4 threads', '8 threads', '16 threads', '32 threads'])
	ax.set_xticklabels(ell_keys, rotation='vertical', fontsize=8)
	ax.set_xticks(np.arange(len(ell_keys)), rotation='vertical', fontsize=8)
	plt.xlabel("Matrice", fontsize=11)
	plt.ylabel("Speedup", fontsize=11)
	plt.title("CPU ELLPACK performance", fontsize=13)
	
	print(ell_keys)

	plt.show()

	print_table(keys, ell_keys, [value[2] for value in values], [value[2] for value in ell_values], "Speedup CPU")

def gpu_istogramma_speedup():

	dictonary = csv.DictReader(open(FILE, newline=''))

	gpu_csr = {}
	gpu_ellpack = {}

	for row in dictonary:
		mode = row['CalculationMode']
		threads = row['Threads']
		if( "-cudaCSR" in mode ):
			#csr
			name = row["Matrix"]
			gpu_serial_time = get_serial_calc_time(name)
			gpu_speedup = float(gpu_serial_time) / float(row["CalculationTime(ms)"])
			matrix = gpu_csr.get(name)
			if ( index := get_block_size(threads)) == -1:
				continue
			if matrix:
				matrix[index] = gpu_speedup
			else:
				matrix = [0,0,0,0]
				matrix[index] = gpu_speedup
			gpu_csr[name] = matrix

		elif( "-cudaELLPACK" in mode ):
			#ellpack
			name = row["Matrix"]
			gpu_serial_time = get_serial_calc_time(name)
			gpu_speedup = float(gpu_serial_time) / float(row["CalculationTime(ms)"])
			matrix = gpu_ellpack.get(name)
			if ( index := get_block_size(threads)) == -1:
				continue
			if matrix:
				matrix[index] = gpu_speedup
			else:
				matrix = [0,0,0,0]
				matrix[index] = gpu_speedup
			gpu_ellpack[name] = matrix

	#graficare

	keys = [key[1:-4] for key in gpu_csr.keys()]
	values = [value for value in gpu_csr.values()]
	fig, ax = plt.subplots()
	bs128 = ax.bar(np.arange(len(keys)) - 0.3, [value[0] for value in values],
	       width=0.2, color='b', align='center')
	bs256 = ax.bar(np.arange(len(keys)) - 0.1,
	       [value[1] for value in values],
	       width=0.2, color='g', align='center')
	bs512 = ax.bar(np.arange(len(keys)) + 0.1,
	       [value[2] for value in values],
	       width=0.2, color='r', align='center')
	bs1024 = ax.bar(np.arange(len(keys)) + 0.3,
	       [value[3] for value in values],
	       width=0.2, color='orange', align='center')
	ax.legend([bs128, bs256, bs512, bs1024], ['128 threads per blocco', '256 threads per blocco', '512 threads per blocco', '1024 threads per blocco'])
	ax.set_xticklabels(keys, rotation='vertical', fontsize=8)
	ax.set_xticks(np.arange(len(keys)), rotation='vertical', fontsize=8)
	
	plt.xlabel("Matrice", fontsize=11)
	plt.ylabel("Speedup", fontsize=11)
	plt.title("GPU CSR performance", fontsize=13)

	plt.show()

	ell_keys = [key[1:-4] for key in gpu_ellpack.keys()]
	ell_values = [value for value in gpu_ellpack.values()]
	fig, ax = plt.subplots()
	bs128 = ax.bar(np.arange(len(ell_keys)) - 0.3, [value[0] for value in ell_values],
	       width=0.2, color='b', align='center')
	bs256 = ax.bar(np.arange(len(ell_keys)) - 0.1,
	       [value[1] for value in ell_values],
	       width=0.2, color='g', align='center')
	bs512 = ax.bar(np.arange(len(ell_keys)) + 0.1,
	       [value[2] for value in ell_values],
	       width=0.2, color='r', align='center')
	bs1024 = ax.bar(np.arange(len(ell_keys)) + 0.3,
	       [value[3] for value in ell_values],
	       width=0.2, color='orange', align='center')
	ax.legend([bs128, bs256, bs512, bs1024], ['128 threads per blocco', '256 threads per blocco', '512 threads per blocco', '1024 threads per blocco'])
	ax.set_xticklabels(ell_keys, rotation='vertical', fontsize=8)
	ax.set_xticks(np.arange(len(ell_keys)), rotation='vertical', fontsize=8)
	
	plt.xlabel("Matrice", fontsize=11)
	plt.ylabel("Speedup", fontsize=11)
	plt.title("GPU ELLPACK performance", fontsize=13)

	plt.show()

	print_table(keys, ell_keys, [value[0] for value in values], [value[0] for value in ell_values], "Speedup GPU")




def cpu_grafico_gflop():
	dictonary = csv.DictReader(open(FILE, newline=''))

	cpu_csr = {}
	cpu_ellpack = {}

	for row in dictonary:
		name = row["Matrix"]
		mode = row['CalculationMode']
		threads = row['Threads']
		if( "-ompCSR" in mode ):
			index = THREADS.index(int(threads))
			gflops = row['GFlops']
			matrix = cpu_csr.get(name)
			if matrix:
				matrix[index] = float(gflops)
			else:
				matrix = [0] * len(THREADS)
				matrix[index] = float(gflops)
			cpu_csr[name] = matrix
		
		elif( "-ompELLPACK" in mode ):
			index = THREADS.index(int(threads))
			gflops = row['GFlops']
			matrix = cpu_ellpack.get(name)
			if matrix:
				matrix[index] = float(gflops)
			else:
				matrix = [0] * len(THREADS)
				matrix[index] = float(gflops)
			cpu_ellpack[name] = matrix

	for key in cpu_csr.keys():
		plt.plot(THREADS, cpu_csr[key], label=key[1:-4])

	plt.xlabel("Threads", fontsize=11)
	plt.ylabel("FLOPS", fontsize=11)
	plt.title("CPU CSR performance", fontsize=13)
	plt.legend(loc = "upper right")

	plt.show()

	for key in cpu_ellpack.keys():
		plt.plot(THREADS, cpu_ellpack[key], label=key[1:-4])

	plt.xlabel("Threads", fontsize=11)
	plt.ylabel("FLOPS", fontsize=11)
	plt.title("CPU ELLPACK performance", fontsize=13)
	plt.legend(loc = "upper right")

	plt.show()

	print_table([key[1:-4] for key in cpu_csr.keys()], [key[1:-4] for key in cpu_ellpack.keys()], [value[int(BEST_TH) - 1] for value in cpu_csr.values()], [value[int(BEST_TH) - 1] for value in cpu_ellpack.values()], "GFlops CPU")



def gpu_grafico_gflop():
	dictonary = csv.DictReader(open(FILE, newline=''))

	gpu_csr = {}
	gpu_ellpack = {}

	for row in dictonary:
		name = row["Matrix"]
		mode = row['CalculationMode']
		blocks = row['Threads']
		if( "-cudaCSR" in mode ):
			index = BLOCKS.index(int(blocks))
			gflops = row['GFlops']
			matrix = gpu_csr.get(name)
			if matrix:
				matrix[index] = float(gflops)
			else:
				matrix = [0] * len(BLOCKS)
				matrix[index] = float(gflops)
			gpu_csr[name] = matrix

		elif( "-cudaELLPACK" in mode ):
			index = BLOCKS.index(int(blocks))
			gflops = row['GFlops']
			matrix = gpu_ellpack.get(name)
			if matrix:
				matrix[index] = float(gflops)
			else:
				matrix = [0] * len(BLOCKS)
				matrix[index] = float(gflops)
			gpu_ellpack[name] = matrix

	for key in gpu_csr.keys():
		plt.plot(BLOCKS, gpu_csr[key], label=key[1:-4])

	plt.xlabel("Block Size", fontsize=11)
	plt.ylabel("FLOPS", fontsize=11)
	plt.title("GPU CSR performance", fontsize=13)
	plt.legend(loc = "upper right")

	plt.show()


	for key in gpu_ellpack.keys():
		plt.plot(BLOCKS, gpu_ellpack[key], label=key[1:-4])

	plt.xlabel("Block Size", fontsize=11)
	plt.ylabel("FLOPS", fontsize=11)
	plt.title("GPU ELLPACK performance", fontsize=13)
	plt.legend(loc = "upper right")
	
	plt.show()

	print_table([key[1:-4] for key in gpu_csr.keys()], [key[1:-4] for key in gpu_ellpack.keys()], [value[0] for value in gpu_csr.values()], [value[0] for value in gpu_ellpack.values()], "GFlops GPU")



def speedup_cpu_gpu():
	dictonary = csv.DictReader(open(FILE, newline=''))

	csr = {}
	ellpack = {}

	for row in dictonary:
		mode = row['CalculationMode']
		threads = row['Threads']
		if( "-omp" in mode and row['Threads'] == BEST_TH):
			#csr
			name = row["Matrix"]
			if "CSR" in mode:
				cuda_csr_time = get_cuda_csr_time(name)
				omp_csr_time = row['CalculationTime(ms)']
				speedup_csr = float(omp_csr_time) / float(cuda_csr_time)
				csr[name] = speedup_csr
			elif "ELLPACK" in mode:
				cuda_ellpack_time = get_cuda_ellpack_time(name)
				omp_ellpack_time = row['CalculationTime(ms)']
				speedup_ellpack = float(omp_ellpack_time) / float(cuda_ellpack_time)
				ellpack[name] = speedup_ellpack

	threshold = 1
	csr_keys = [key[1:-4] for key in csr.keys()]
	plt.axhline(y=threshold, linewidth=1, color='r')

	plt.bar(csr_keys, csr.values(), color="darkcyan")

	plt.xticks(np.arange(len(csr)), rotation='vertical', fontsize=8)
	plt.ylabel("Speedup cpu-gpu", fontsize=11)
	plt.xlabel("Matrice", fontsize=11)
	plt.title("Confronto performance CPU-GPU formato CSR", fontsize=13)

	plt.show()

	ell_keys = [key[1:-4] for key in ellpack.keys()]
	plt.axhline(y=threshold, linewidth=1, color='r')

	plt.bar(ell_keys, ellpack.values(), color="darkcyan")

	plt.xticks(np.arange(len(ellpack)), rotation='vertical', fontsize=8)
	plt.ylabel("Speedup cpu-gpu", fontsize=11)
	plt.xlabel("Matrice", fontsize=11)
	plt.title("Confronto performance CPU-GPU formato ELLPACK", fontsize=13)

	plt.show()

cpu_istogramma_speedup()
gpu_istogramma_speedup()

cpu_grafico_gflop()
gpu_grafico_gflop()

speedup_cpu_gpu()