#' Copy WorldPop folder to your local working directory. 
#' @description Copy files from WorldPop server to your local working directory. 
#' This allows everyone to work locally with identical inputs.
#' @param srcdir characer. Directory of source files
#' @param outdir character. Directory to write output files to
#' @param OS.type character ('windows' or 'unix'). Type of operating system (see \code{.Platform$OS.type})
#' @param md5check logical. If TRUE, an md5sum check is performed to compare contents of source files to local files. If they differ, the local files will be overwritten.
#' @return Writes input files to disk
#' @export

copyWP <- function(srcdir, outdir, OS.type='windows', md5check=FALSE){

  # format server name
  if(.Platform$OS.type=="windows"){
    srcdir <- file.path('//worldpop.files.soton.ac.uk/worldpop', srcdir)
  }
  if(.Platform$OS.type=="unix"){
    srcdir <- file.path('/Volumes/worldpop', srcdir)
  } 

  # check source directory exists
  if(!dir.exists(srcdir)){
    
    stop(paste('Source directory does not exist:', srcdir))
    
  } else {
    
    # create output directory
    if(!dir.exists(outdir)) dir.create(outdir, showWarnings=F)

    # list directories
    ld <- list.dirs(srcdir, full.names=F)
    
    # create directories
    for(d in ld[-1]){
      dir.create(file.path(outdir, d), showWarnings=F, recursive=T)
    }
    
    # list files
    lf <- list.files(srcdir, recursive=T, include.dirs=F)

    # copy files
    for(f in lf){
      
      # check if file needed
      toggleCopy <- !file.exists(file.path(outdir,f))
      
      # md5 check
      if(!toggleCopy & md5check){

        # check file contents
        md5src <- as.character(tools::md5sum(file.path(srcdir,f)))
        md5out <- as.character(tools::md5sum(file.path(outdir,f)))

        if(!md5src==md5out){
          toggleCopy <- T
        }
      }
      
      # copy the file if needed
      if(toggleCopy){
        cat(paste0(f,'\n'))
        file.copy(from=file.path(srcdir,f), to=file.path(outdir,f), overwrite=T)
      }
    }
  }
}
