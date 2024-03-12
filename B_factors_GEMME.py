# %%
import sys
from prody import *
import numpy as np
import pandas as pd
import os
import re
import scipy.stats as ss
from scipy.stats import pearsonr
import pymol
import argparse

# %% [markdown]
# ## We have 13 458 structures predicted by AlphaFold

# %%
## pattern to get UnoProt ID from AF file name 
patternAF = re.compile('(?<=AF-)[A-Z 0-9]*')
# %%
def parse_command_line():
    """
    Parse command line.
    """
    parser = argparse.ArgumentParser(
        prog="B_factors_GEMME.py",
        description="""Change b factors from AlphaFold confidence level
            to mean or max GEMME value for a given residue; 
            the Ensembl_UniProt_FB.txt correspondance file should be in the same folder than this script """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## we work either with FBid and the two directories with GEMME predictions and AF structures
    parser.add_argument('-FBid',
                        help="Input FlyBase ID if you do not have the structure yet; "
                        "the program will seek for the UniProt ID and find the corresponding 3D structure in the provided path -AFdir",
                        type=str) 

    parser.add_argument('-afdir',
                        help='Directory with all 3D structures from AlphaFoldDB; mandatory if working with only FlyBase ID; By default current directory',
                        type=str,
                        default='') ## pas sure du default './AF_Fly/'

    parser.add_argument('-gdir',
                        help='Input directory. By default current directory',
                        type=str,
                        default='')#/CombiPred/
    ## either with 3D structure and GEMME Matrix
    parser.add_argument('-mat',
                        help='GEMME matrix; if you want to work directly with GEMME matrix; You should also provide the 3D structure',
                        type=str,
                        )   #default='.'

    parser.add_argument('-s',
                        '--struct',
                        help='3D structure with AF confidence level',
                        type=str,
                        default='')
    ## if you want to soecify the output directory
    parser.add_argument('-o', '--outdir',
                        help='Output directory. By default the current directory ',
                        type=str,
                        default='')
    ## choose max or mean of each column
    parser.add_argument('-col',
                        help='Options: "mean" or "max". Take the average or maximum of the column in the GEMME matrix for each residue. ',
                        type=str,
                        default='max')
    parser.add_argument('-val',
                        help='Options: "raw" or "rank". Take the raw values of the GEMME matrix or thier ranks. For both option minimum is neutral, maximum is impactful.',
                        type=str,
                        default='raw')
    parser.add_argument('-png',
                        help='Options: 0 or 1. If you want to get 2 png of the 3D structure with AF and GEMME colors put 1.'
                             '0 by default.',
                        type=int,
                        default=0)
  

    return parser

# %%
def GemmeMat_Rank(dir_name, pp_name, option):
    
    '''read GEMME matrix;and return matrix with ranks '''

    mat = np.loadtxt(dir_name+pp_name, dtype='str', skiprows=1)
    mat=np.delete(mat, 0, axis=1)                                       ## we delete the letters at the beginning of each matrix's row 
    mat[mat=='NA']=0.0
    mat = mat.astype(float)
    print("Protein length", mat.shape[1], np.count_nonzero(mat == 0.0))
    ## from raw values to ranks
    if mat.min().min()==0.0:
        sys.exit('bad jet execution, only 0')
    print('Type of values', option)
    if option=='rank':
        mat_rang = np.array(ss.rankdata(mat))
        mat_rang = mat_rang/len(mat_rang)
        mat_out = (1 - mat_rang.reshape(20, mat.shape[1]))*100
    elif option=='raw':
        mat_out=-mat
    
    return mat_out

# %%
def changeBfactors(AF_file, mat_GEMME, func, GEMME_PDB):
    #Parse the protein structure and get Alpha Fold b factors 
    structure = parsePDB(AF_file)
    calphas = structure.select('name CA')
    N = calphas.numAtoms()
    bfactor = calphas.getBetas()
    print('AF bfactors limits', min(bfactor), max(bfactor))

    ## check that the number of calpha and the matrix's shape are the same
    if N != mat_GEMME.shape[1]:
        sys.exit(f"ERROR: the structure does not correspond to the GEMME prediction")

    ## we take the better correlated with AF's bfactors 
    vecMax_GEMME = mat_GEMME.max(axis=0)
    vecMean_GEMME = mat_GEMME.mean(axis=0)
    
    New_Bfactors = vecMax_GEMME if func=='max' else vecMean_GEMME
    print(func, min(New_Bfactors), max(New_Bfactors))
    # New_Bfactors = vecMax_GEMME if pearsonr(vecMax_GEMME,bfactor) > pearsonr(vecMean_GEMME,bfactor) else vecMean_GEMME

    ##  generate a PDB file with gemme values as bfactors
    calphas.setBetas(New_Bfactors)
    print(','.join([str(x) for x in New_Bfactors]))
    writePDB(GEMME_PDB+'_GEMME.pdb', calphas)
 
#%% 
def main():
    
    args = parse_command_line().parse_args()

    if len(sys.argv) == 1:
        parse_command_line().print_help()
        sys.exit('ERROR: there is no enough poassed arguments')

    print(args)
    

    ## get UniProt id and generate the corresponding file name
    ## see if the Uniprot ID exists, then 3D structure
    if args.outdir:
        if not os.path.exists(args.outdir):
            os.mkdir(args.outdir)

    if args.FBid:
        UniProt_FBpp = pd.read_csv('Ensembl_UniProt_FB.txt', index_col='FlyBase translation ID')
        try:
            UniProt_id = UniProt_FBpp.loc[args.FBid, 'UniProtKB/TrEMBL ID']
            print(UniProt_id)
            AF_file=args.afdir+f'AF-{UniProt_id}-F1-model_v4.pdb.gz'
    
        except:
            sys.exit("ERROR: no correspondance found for UniPort ID; try to look in FlyBase AlphaFold DB and give the structure")

        matGEMME=GemmeMat_Rank(args.gdir,args.FBid+'_normPred_evolCombi.txt', args.val)
        GEMME_PDB = args.outdir+args.FBid
    
    
    ## GEMME matrix is provided
    ## mat={FBpp_id}_normPred_evolCombi.txt
    if args.mat:
        print("working with the provided matrix", args.mat)
        matGEMME=GemmeMat_Rank(args.gdir,args.mat, args.val)
        args.FBid = args.mat[:11]
        print(args.FBid)
        AF_file = args.afdir+args.struct
        GEMME_PDB = args.outdir+args.FBid


    if not os.path.exists(AF_file):
        sys.exit("ERROR: there is no 3D structure in AF_Fly/ or given folder; Try to look in AF and give the 3D structure to this script; or wrong directory")
    
    changeBfactors(AF_file, matGEMME, args.col, GEMME_PDB) 
    
    ## load both files 
    if args.png:
        ## GEMME
        pymol.cmd.load(GEMME_PDB+'_GEMME.pdb')
        pymol.cmd.spectrum('b', 'red_white_blue', 'all')
        #pymol.cmd.unset('opaque_background')
        #pymol.cmd.bg_color('grey')
        pymol.cmd.png(GEMME_PDB+"_GEMME.png", dpi=900)
        pymol.cmd.delete(args.FBid+'_GEMME')
        ## AF 
        pymol.cmd.load(AF_file)
        pymol.cmd.spectrum('b', 'red_white_blue', 'all')
        pymol.cmd.bg_color('grey')
        pymol.cmd.png(GEMME_PDB+"_AF.png", dpi=1600)
        pymol.cmd.quit()


main()
