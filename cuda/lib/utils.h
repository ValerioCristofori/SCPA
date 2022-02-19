
struct matrix{
	int 			*I;  	// row index
	int 			*J;  	// column index
	double 			*val;	// array values
	int 			nz;		// number of non-zeros
	int 			M;		// number of rows
	int 			N;      // number of cols
};


struct vector{
	int 			xdim;   // array length
	double 			*X; 	// array values
};



struct Csr{
	int    	 *IRP;		// vector of pointers at the beginning of each line
	int   	 *JA;		// column index vector
	double 	 *AS;		// coefficients vector
	int 	  M;		// number of rows
	int 	  N;		// number of cols
	int 	  nz;		// number of non-zeros
};


struct Ellpack{
	int   	   *JA_t;	// column index vector: transposed matrix
	double 	   *AS_t;	// coefficients vector: transposed matrix
	int 	  	M;		// number of rows
	int 	  	N;		// number of columns
	int 		nz;		// number of non-zeros
	int 		maxnz;	// the maximum of non-zeros per row
	int        *MAXNZ;	// vector of non-zero num each row
};

struct Result{
	double     *res;			// result array
	int 		len;			// length of the array
	double     	elapsed_time;	// time in calculation
	double 	 	gpuflops;		// floating point ops per sec
};


// ---------------- utils ------------------------

void getmul(struct matrix *mat, struct vector *vec, double* res);

void quicksort(double* a, double* vindex, int* rindex, int* cindex, int n);

void dprintArrayInt(int* a, int len);

void dprintArrayDouble(double* a, int len);


// ------------------ pre processing ---------------

struct Csr* preprocess_csr(struct matrix *mat);

struct Ellpack* preprocess_ellpack(struct matrix *mat);



// ------------------ calculation -----------------

int load_matrix(char *matrix_filename, struct matrix* mat);

int load_vector(char *vector_filename, struct vector* vec, int M);

int calculate_prod(struct matrix *mat, struct vector* vec, double *res_seq, char* mode, FILE *fpt);

