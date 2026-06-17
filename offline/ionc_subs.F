c**************************************************************************
c
c     This module is used to read and save netCDF files for performing
c     the program coarse_grain.
c
c     Written by Long Li, 10 Aug. 2020.  
c      
c**************************************************************************      
c
      MODULE ionc_subs

      IMPLICIT NONE

      PRIVATE

*     Subroutines      
      PUBLIC :: read_ini, save_ini, read_out, save_out, handle_err

*     Storage for identifiers
      integer, PUBLIC, SAVE :: incid, oncid
      integer, PRIVATE, SAVE :: pid, pfid, prid, ufid, urid, vfid, vrid

      CONTAINS

c**************************************************************************      
c
      SUBROUTINE read_ini

*     Modules
      USE param, ONLY : nxpo,nypo,nlo,nto    
      USE state, ONLY : tyrs,xpo,ypo,zo

      IMPLICIT NONE
      INCLUDE 'netcdf.inc'

*     Local parameters
      character (len=*), parameter :: subnam = 'read_ini'
      integer ncstat,varid
      real ttmp(nto),xtmp(nxpo),ytmp(nypo),ztmp(nlo)

*     Read time axis (yrs)
      ncstat = nf_inq_varid (incid, 'time', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var (incid, varid, ttmp)
      tyrs = dble(ttmp)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

*     Read p grid x-axis (km) 
      ncstat = nf_inq_varid (incid, 'xp', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var (incid, varid, xtmp)
      xpo = dble(xtmp)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

*     Read p-grid y-axis (km)     
      ncstat = nf_inq_varid (incid, 'yp', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var (incid, varid, ytmp)
      ypo = dble(ytmp)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
     
*     Read z-axis (km)     
      ncstat = nf_inq_varid (incid, 'z', varid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_get_var (incid, varid, ztmp)
      zo = dble(ztmp)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_inq_varid (incid, 'p', pid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      END SUBROUTINE read_ini
c
c**************************************************************************      
c
      SUBROUTINE read_out (p, idt)

*     Modules
      USE param, ONLY : nxpo,nypo,nlo    

      IMPLICIT NONE
      INCLUDE 'netcdf.inc'

*     I/O arguments
      double precision, INTENT(OUT) :: p(nxpo,nypo,nlo)
      integer, INTENT(IN) :: idt

*     Local parameters
      character (len=*), parameter :: subnam = 'read_out'
      integer ncstat,starts(4),counts(4)
      real ptmp(nxpo,nypo,nlo)

*     Read eddy-resolving pressure (m^2/s^2)
      starts(1) = 1
      starts(2) = 1
      starts(3) = 1
      starts(4) = idt
      counts(1) = nxpo
      counts(2) = nypo
      counts(3) = nlo
      counts(4) = 1
      ncstat = nf_get_vara_real (incid, pid, starts, counts, ptmp)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      p = dble(ptmp)

      END SUBROUTINE read_out
c
c**************************************************************************      
c
      SUBROUTINE save_ini

*     Modules      
      USE param, ONLY : nto,nxpoc,nypoc,nxtoc,nytoc,nlo    
      USE state, ONLY : xpoc,ypoc,xtoc,ytoc,zo,tyrs 

      IMPLICIT NONE
      INCLUDE 'netcdf.inc'

*     Local parameters
      character (len=*), parameter :: subnam = 'save_ini'
      integer ncstat,tdim,xpdim,ypdim,xtdim,ytdim,zdim,tid,
     &        xpid,ypid,xtid,ytid,zid,dims(4)

*     Define dimensions
      ncstat = nf_def_dim (oncid, 'time', nto, tdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_def_dim (oncid, 'xp', nxpoc, xpdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_def_dim (oncid, 'yp', nypoc, ypdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_def_dim (oncid, 'xt', nxtoc, xtdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_def_dim (oncid, 'yt', nytoc, ytdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_def_dim (oncid, 'z', nlo, zdim)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

*     Define 1D variables with attributes
      ncstat = nf_def_var (oncid, 'time', NF_DOUBLE, 1, tdim, tid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, tid, 'long_name', 9, 'Time axis')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, tid, 'units', 5, 'years')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      
      ncstat = nf_def_var (oncid, 'xp', NF_DOUBLE, 1, xpdim, xpid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, xpid, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, xpid, 'long_name', 15, 
     &                          'X axis (p-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      
      ncstat = nf_def_var (oncid, 'yp', NF_DOUBLE, 1, ypdim, ypid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, ypid, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, ypid, 'long_name', 15, 
     &                          'Y axis (p-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      
      ncstat = nf_def_var (oncid, 'xt', NF_DOUBLE, 1, xtdim, xtid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, xtid, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, xtid, 'long_name', 15, 
     &                          'X axis (T-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      
      ncstat = nf_def_var (oncid, 'yt', NF_DOUBLE, 1, ytdim, ytid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, ytid, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, ytid, 'long_name', 15, 
     &                          'Y axis (T-grid)')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_def_var (oncid, 'z', NF_DOUBLE, 1, zdim, zid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, zid, 'units', 2, 'km')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, zid, 'long_name', 20,
     &                          'Mid-layer depth axis')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
 
*     Define 4D variables with attributes
      dims(1) = xpdim 
      dims(2) = ypdim
      dims(3) = zdim
      dims(4) = tdim 
      ncstat = nf_def_var (oncid, 'pf', NF_DOUBLE, 4, dims, pfid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, pfid, 'units', 7, 'm^2/s^2')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, pfid, 'long_name', 23,
     &                          'Coarse-grained pressure')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      
      ncstat = nf_def_var (oncid, 'pr', NF_DOUBLE, 4, dims, prid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, prid, 'units', 7, 'm^2/s^2')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, prid, 'long_name', 17,
     &                          'Residual pressure')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      
      dims(1) = xpdim 
      dims(2) = ytdim
      dims(3) = zdim
      dims(4) = tdim 
      ncstat = nf_def_var (oncid, 'uf', NF_DOUBLE, 4, dims, ufid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, ufid, 'units', 3, 'm/s')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, ufid, 'long_name', 29,
     &                          'Coarse-grained zonal velocity')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_def_var (oncid, 'ur', NF_DOUBLE, 4, dims, urid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, urid, 'units', 3, 'm/s')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, urid, 'long_name', 23,
     &                          'Residual zonal velocity')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      
      dims(1) = xtdim 
      dims(2) = ypdim
      dims(3) = zdim
      dims(4) = tdim 
      ncstat = nf_def_var (oncid, 'vf', NF_DOUBLE, 4, dims, vfid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, vfid, 'units', 3, 'm/s')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, vfid, 'long_name', 34,
     &                          'Coarse-grained meridional velocity')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      ncstat = nf_def_var (oncid, 'vr', NF_DOUBLE, 4, dims, vrid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, vrid, 'units', 3, 'm/s')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_att_text (oncid, vrid, 'long_name', 28,
     &                          'Residual meridional velocity')
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

*     Leave definition mode and entering data mode
      ncstat = nf_enddef (oncid)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

*     Write 1D data to variables
      ncstat = nf_put_vara_double (oncid, tid, 1, nto, tyrs)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_vara_double (oncid, xpid, 1, nxpoc, xpoc)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_vara_double (oncid, ypid, 1, nypoc, ypoc)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_vara_double (oncid, xtid, 1, nxtoc, xtoc)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_vara_double (oncid, ytid, 1, nytoc, ytoc)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_vara_double (oncid, zid, 1, nlo, zo)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      
      END SUBROUTINE save_ini
c
c**************************************************************************      
c
      SUBROUTINE save_out (idt)

*     Modules      
      USE param, ONLY : nxpoc,nypoc,nxtoc,nytoc,nlo    
      USE state, ONLY : pf,pr,uf,ur,vf,vr 

      IMPLICIT NONE
      INCLUDE 'netcdf.inc'

*     I/O arguments
      integer, INTENT(IN) :: idt

*     Local parameters
      character (len=*), parameter :: subnam = 'save_out'
      integer ncstat,starts(4),counts(4)

*     Store pf and pr
      starts(1) = 1
      starts(2) = 1
      starts(3) = 1
      starts(4) = idt
      counts(1) = nxpoc
      counts(2) = nypoc
      counts(3) = nlo
      counts(4) = 1
      ncstat = nf_put_vara_double (oncid, pfid, starts, counts, pf)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_vara_double (oncid, prid, starts, counts, pr)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      
*     Store uf and ur      
      starts(1) = 1
      starts(2) = 1
      starts(3) = 1
      starts(4) = idt
      counts(1) = nxpoc
      counts(2) = nytoc
      counts(3) = nlo
      counts(4) = 1
      ncstat = nf_put_vara_double (oncid, ufid, starts, counts, uf)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_vara_double (oncid, urid, starts, counts, ur)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

*     Store vf and vr      
      starts(1) = 1
      starts(2) = 1
      starts(3) = 1
      starts(4) = idt
      counts(1) = nxtoc
      counts(2) = nypoc
      counts(3) = nlo
      counts(4) = 1
      ncstat = nf_put_vara_double (oncid, vfid, starts, counts, vf)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)
      ncstat = nf_put_vara_double (oncid, vrid, starts, counts, vr)
      if ( ncstat.ne.NF_NOERR ) call handle_err (ncstat, subnam)

      END SUBROUTINE save_out
c      
c**************************************************************************
c
      SUBROUTINE handle_err (ncstat, fromst)

      IMPLICIT NONE
      INCLUDE 'netcdf.inc'

*     Subroutine arguments
      integer, INTENT(IN) :: ncstat
      character (len=*), INTENT(IN), OPTIONAL :: fromst

*     fromst is an optional string indicating where the call came
*     from that caused the netCDF problem (e.g. subroutine name).

*     Routine which interprets errors from netCDF output functions,
*     prints them to standard output and then kills the whole run.
      if ( ncstat.ne.NF_NOERR ) then
        if ( present(fromst) ) then
          print *, trim(fromst)//':'//trim( nf_strerror(ncstat) )
         else
          print *, trim( nf_strerror(ncstat) )
        endif
        stop 'netCDF:: STOPPED'
      endif

      END SUBROUTINE handle_err
c      
c**************************************************************************
c
      END MODULE ionc_subs
c      
c**************************************************************************
