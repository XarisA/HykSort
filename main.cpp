#include <iostream>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cassert>
#include <algorithm>
#include "functions.h"

#define N 16 //Number of elements to sort N

std::vector<unsigned int> HykSort(std::vector<unsigned int> arr, unsigned  int kway, MPI_Comm comm_);
std::vector<unsigned int> ParallelSelect(std::vector<unsigned int>& arr, unsigned int kway, MPI_Comm comm);


int main(int argc , char *argv[]) {

    int p; // Number of MPI tasks. Currently running with argument -np 4
    int rank;
    int tag=10;
    int i,j,l;
    std::vector<unsigned int> A; //Input array .. Check if I need a pointer
    std::vector<unsigned int> B,Bl;

    unsigned int k = 4; //Number of splitters


    MPI_Comm comm;
    MPI_Status status;
    comm = MPI_COMM_WORLD;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    int n = (N / p); //Number of elements in each process
    std::vector<unsigned int> Ar(n); //Array to be sorted local block

    //Create my Array
    Ar = init_array(n, rank);
    //Print my array
//    print_array_in_process(Ar, n, p, rank);

    MPI_Barrier(comm);  //Wait for previous sends/receives to finish

    B = HykSort(Ar, k, comm);

    MPI_Barrier(comm);

    print_array_in_process(B, B.size(),p,rank);

    MPI_Barrier(comm);

    MPI_Gather(&B, B.size(), MPI_INT, &A, B.size(), MPI_INT, 0, MPI_COMM_WORLD);
    //print_array(A, n);

//    if (rank != 0) {
//        A.insert(std::end(A), std::begin(Bl), std::end(Bl));
//    }

    MPI_Finalize();
    return 0;
}
std::vector<unsigned int> ParallelSelect(std::vector<unsigned int>& arr, unsigned int kway, MPI_Comm comm) {
    int rank, npes;
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &rank);

    //-------------------------------------------
    int totSize, nelem = arr.size();
    MPI_Allreduce(&nelem, &totSize, 1,MPI_INT, MPI_SUM, comm);

    //Determine splitters. O( log(N/p) + log(p) )
    int splt_count = (1000*kway*nelem)/totSize;
    if (npes>1000*kway) splt_count = (((float)rand()/(float)RAND_MAX)*totSize<(1000*kway*nelem)?1:0);
    if (splt_count>nelem) splt_count=nelem;
    std::vector<unsigned int> splitters(splt_count);
    for(size_t i=0;i<splt_count;i++)
        splitters[i] = arr[rand()%nelem];

    // Gather all splitters. O( log(p) )
    int glb_splt_count;
    std::vector<int> glb_splt_cnts(npes);
    std::vector<int> glb_splt_disp(npes,0);
    MPI_Allgather(&splt_count,1,MPI_INT, &glb_splt_cnts[0],1,MPI_INT, comm);
    scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);
    glb_splt_count = glb_splt_cnts[npes-1] + glb_splt_disp[npes-1];
    std::vector<unsigned int> glb_splitters(glb_splt_count);
    MPI_Allgatherv(&    splitters[0], splt_count, MPI_INT,
                   &glb_splitters[0], &glb_splt_cnts[0], &glb_splt_disp[0],
                   MPI_INT, comm);

    // rank splitters. O( log(N/p) + log(p) )
    std::vector<int> disp(glb_splt_count,0);
    if(nelem>0){
#pragma omp parallel for
        for(size_t i=0; i<glb_splt_count; i++){
            disp[i] = std::lower_bound(&arr[0], &arr[nelem], glb_splitters[i]) - &arr[0];
        }
    }
    std::vector<int> glb_disp(glb_splt_count, 0);
    MPI_Allreduce(&disp[0], &glb_disp[0], glb_splt_count, MPI_INT, MPI_SUM, comm);

    std::vector<unsigned int> split_keys(kway);
#pragma omp parallel for
    for (unsigned int qq=0; qq<kway; qq++) {
        int* _disp = &glb_disp[0];
        int optSplitter = ((qq+1)*totSize)/(kway+1);
        // if (!rank) std::cout << "opt " << qq << " - " << optSplitter << std::endl;
        for(size_t i=0; i<glb_splt_count; i++) {
            if(labs(glb_disp[i] - optSplitter) < labs(*_disp - optSplitter)) {
                _disp = &glb_disp[i];
            }
        }
        split_keys[qq] = glb_splitters[_disp - &glb_disp[0]];
    }

    return split_keys;
}

std::vector<unsigned int> HykSort(std::vector<unsigned int> arr, unsigned int kway, MPI_Comm comm_) {

    // Copy communicator.
    MPI_Comm comm=comm_;

    int omp_p=omp_get_max_threads();


    // Get comm size and rank.
    int npes, myrank;
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &myrank);
    srand(myrank);


    // Local and global sizes. O(log p)
    size_t totSize =0;
    size_t nelem=arr.size();
    MPI_Allreduce(&nelem, &totSize, 1,MPI_INT,MPI_SUM, comm);
/*
    As an alternative I can use the followin
    MPI_Reduce(&nelem, &totSize, 1,MPI_INT,MPI_SUM,0, comm);
    MPI_Bcast(&totSize,1,MPI_INT,0,comm);
*/
    std::vector<unsigned int> arr_(nelem*2); //Extra buffer.
    std::vector<unsigned int> arr__(nelem*2); //Extra buffer.

    // Local sort.
    omp_par::merge_sort(&arr[0], &arr[arr.size()]);

    while(npes>1 && totSize>0){
        if(kway>npes) kway = npes;
        int blk_size=npes/kway; assert(blk_size*kway==npes);
        int blk_id=myrank/blk_size, new_pid=myrank%blk_size;

        // Determine splitters.
        std::vector<unsigned int> split_key = ParallelSelect(arr, kway-1, comm);

        {// Communication

            // Determine send_size.
            std::vector<int> send_size(kway), send_disp(kway+1); send_disp[0]=0; send_disp[kway]=arr.size();
            for(int i=1;i<kway;i++) send_disp[i]=std::lower_bound(&arr[0], &arr[arr.size()], split_key[i-1])-&arr[0];
            for(int i=0;i<kway;i++) send_size[i]=send_disp[i+1]-send_disp[i];

            // Get recv_size.
            int recv_iter=0;
            std::vector<unsigned int*> recv_ptr(kway);
            std::vector<size_t> recv_cnt(kway);
            std::vector<int> recv_size(kway), recv_disp(kway+1,0);
            for(int i_=0;i_<=kway/2;i_++){
                int i1=(blk_id+i_)%kway;
                int i2=(blk_id+kway-i_)%kway;
                MPI_Status status;
                for(int j=0;j<(i_==0 || i_==kway/2?1:2);j++){
                    int i=(i_==0?i1:((j+blk_id/i_)%2?i1:i2));
                    int partner=blk_size*i+new_pid;
                    MPI_Sendrecv(&send_size[     i   ], 1, MPI_INT, partner, 0,
                                 &recv_size[recv_iter], 1, MPI_INT, partner, 0, comm, &status);
                    recv_disp[recv_iter+1]=recv_disp[recv_iter]+recv_size[recv_iter];
                    recv_ptr[recv_iter]=&arr_[recv_disp[recv_iter]];
                    recv_cnt[recv_iter]=recv_size[recv_iter];
                    recv_iter++;
                }
            }

            // Communicate data.
            int asynch_count=2;
            recv_iter=0;
            int merg_indx=2;
            std::vector<MPI_Request> reqst(kway*2);
            std::vector<MPI_Status> status(kway*2);
            arr_ .resize(recv_disp[kway]);
            arr__.resize(recv_disp[kway]);
            for(int i_=0;i_<=kway/2;i_++){
                int i1=(blk_id+i_)%kway;
                int i2=(blk_id+kway-i_)%kway;
                for(int j=0;j<(i_==0 || i_==kway/2?1:2);j++){
                    int i=(i_==0?i1:((j+blk_id/i_)%2?i1:i2));
                    int partner=blk_size*i+new_pid;

                    if(recv_iter-asynch_count-1>=0) MPI_Waitall(2, &reqst[(recv_iter-asynch_count-1)*2], &status[(recv_iter-asynch_count-1)*2]);
                    MPI_Irecv(&arr_[recv_disp[recv_iter]], recv_size[recv_iter], MPI_INT,partner, 1, comm, &reqst[recv_iter*2+0]);
                    MPI_Issend(&arr [send_disp[     i   ]], send_size[     i   ], MPI_INT,partner, 1, comm, &reqst[recv_iter*2+1]);
                    recv_iter++;

                    int flag[2]={0,0};
                    if(recv_iter>merg_indx) MPI_Test(&reqst[(merg_indx-1)*2],&flag[0],&status[(merg_indx-1)*2]);
                    if(recv_iter>merg_indx) MPI_Test(&reqst[(merg_indx-2)*2],&flag[1],&status[(merg_indx-2)*2]);
                    if(flag[0] && flag[1]){
                        unsigned int* A=&arr_[0]; unsigned int* B=&arr__[0];
                        for(int s=2;merg_indx%s==0;s*=2){
                            //std    ::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                            //               &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]]);
                            omp_par::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                                           &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]],omp_p,std::less<unsigned int>());
                            unsigned int* C=A; A=B; B=C; // Swap
                        }
                        merg_indx+=2;
                    }
                }
            }
            // Merge remaining parts.
            while(merg_indx<=(int)kway){
                MPI_Waitall(1, &reqst[(merg_indx-1)*2], &status[(merg_indx-1)*2]);
                MPI_Waitall(1, &reqst[(merg_indx-2)*2], &status[(merg_indx-2)*2]);
                {
                    unsigned int* A=&arr_[0]; unsigned int* B=&arr__[0];
                    for(int s=2;merg_indx%s==0;s*=2){
                        //std    ::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                        //               &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]]);
                        omp_par::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                                       &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]],omp_p,std::less<unsigned int>());
                        unsigned int* C=A; A=B; B=C; // Swap
                    }
                    merg_indx+=2;
                }
            }
            {// Swap buffers.
                int swap_cond=0;
                for(int s=2;kway%s==0;s*=2) swap_cond++;
                if(swap_cond%2==0) swap(arr,arr_);
                else swap(arr,arr__);
            }
        }

        {// Split comm. kway  O( log(p) ) ??
            MPI_Comm scomm;
            MPI_Comm_split(comm, blk_id, myrank, &scomm );
            if(comm!=comm_) MPI_Comm_free(&comm);
            comm = scomm;

            MPI_Comm_size(comm, &npes);
            MPI_Comm_rank(comm, &myrank);
        }

    }
    return arr;
}


	

