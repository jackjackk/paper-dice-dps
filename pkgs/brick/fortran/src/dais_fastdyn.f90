!=================================================================================
!  Subroutines to run DAIS:
!  simple model for Antarctic ice-sheet radius/volume [m sle] (Schaffer 2014)
!   Original coded by Alexander Bakker
!   WAIS/fast dynamics added by Tony Wong
!======================================================================================
!
! Private parameters/variables globally used within module
!
!   tstep     time step
!
!   b0        Undisturbed bed height at the continent center [m]
!   slope     Slope of ice sheet bed before loading [-]
!   mu        Profile parameter for parabolic ice surface (related to ice stress) [m0.5]
!   h0        hr(Ta=0): Height of runoff line at Ta = 0 [m]
!   c         Sensitivity of Height of runoff line (hr) [m/degC]
!   P0        P(Ta=0): Annual precipitation for Ta = 0 [m (ice equivalent)]
!   kappa     Coefficient for the exponential dependency of precipitation on Ta [degC-1]
!   nu        Proportionality constant relating runoff decrease with height to
!             precipitation [m^(-1/2) yr^(-1/2)]
!   f0        Proportionality constant for ice flow at grounding line [m/yr]
!   gamma     Power for the relation of ice flow speed to water depth [-]
!   alpha     Partition parameter for effect of ocean subsurface temperature on ice flux [-]
!   Toc_0     Present-day, high latitude ocean subsurface temperature [degC]
!   Rad0      Reference ice sheet radius [m]
!   dSLais    logical that tells if < dSL > represents contribution of
!              - all components (including AIS)        -> dSLais = 1
!              - all otherm components (excluding AIS) -> dSLais = 0
!   lf        Mean AIS fingerprint at AIS shore
!
!   Tf        Freecing temperature sea water [degC]
!   ro_w      (Sea) water density [kg/m3]
!   ro_i      Ice density [kg/m3]
!   ro_m      Rock density [kg/m3]
!   Aoc       Surface of the ocean [m2]
!   Tcrit     Trigger temperature, at which disintegration occurs (deg C) (WAIS/fast dynamics parameter)
!   lambda    disintegration rate (m/yr)
!   Vmin      minimum volume, below which there is no more ice to disintegrate (m^3)
!==============================================================================

module dais

    USE global
    implicit none
    private

! parameters:
    real(DP) :: tstep

    real(DP) :: b0
    real(DP) :: slope
    real(DP) :: mu
    real(DP) :: h0
    real(DP) :: c
    real(DP) :: P0
    real(DP) :: kappa
    real(DP) :: nu
    real(DP) :: f0
    real(DP) :: gamma
    real(DP) :: alpha
    real(DP) :: Tf
    real(DP) :: Tcrit
    real(DP) :: lambda
    real(DP) :: Vmin

    real(DP) :: Toc_0
    real(DP) :: Rad0
    real(DP) :: Aoc
    real(DP) :: includes_dSLais
    real(DP) :: lf

    real(DP) :: del
    real(DP) :: eps1
    real(DP) :: eps2

! variables
    real(DP) :: R       ! Radius ice sheet
    real(DP) :: V       ! Volume ice sheet

! public subroutines
    public :: dais_step, init_dais


contains


!------------------------------------------------------------------------------
subroutine init_dais(time_step, parameters, SL, L_Fast_Dynamics, Rad, Vol, disint)
!  =========================================================================
! |  Initialize the DAIS parameters and initial variables.                                   |
!  =========================================================================

    real(DP), intent(IN) :: time_step
    real(DP), dimension(23), intent(IN) :: parameters
    real(DP), intent(IN)  :: SL
    logical , intent(IN)  :: L_Fast_Dynamics
    real(DP), intent(OUT) :: Rad
    real(DP), intent(OUT) :: Vol
    real(DP), intent(OUT) :: disint

    real(DP) :: rc, rho_w, rho_i, rho_m


! Assign values to model parameters
    tstep  = time_step
    b0     = parameters(1)
    slope  = parameters(2)
    mu     = parameters(3)
    h0     = parameters(4)
    c      = parameters(5)
    P0     = parameters(6)
    kappa  = parameters(7)
    nu     = parameters(8)
    f0     = parameters(9)
    gamma  = parameters(10)
    alpha  = parameters(11)
    Tf     = parameters(12)
    rho_w  = parameters(13)
    rho_i  = parameters(14)
    rho_m  = parameters(15)
    Toc_0  = parameters(16)
    Rad0   = parameters(17)
    Aoc    = parameters(18)
    lf     = parameters(19)
    includes_dSLais = parameters(20)
    if(L_Fast_Dynamics) then
      Tcrit  = parameters(21)
      lambda = parameters(22)
      Vmin   = parameters(23)
    else
      Tcrit  = 999.999
      lambda = 0.0
      Vmin   = -999.999
    end if

! Initialize intermediate parameters
    del  = rho_w / rho_i
    eps1 = rho_i /(rho_m - rho_i)
    eps2 = rho_w /(rho_m - rho_i)

! Initial values
    R      = Rad0
    rc     = (b0 - SL)/slope
    V      = Pi * (1+eps1) * ( (8./15.) * mu**0.5 * R**2.5 - (1./3.)*slope*R**3)
    if(R > rc) then
       V   = V - Pi*eps2 * ( (2./3.)  * slope*(R**3-rc**3)-b0*(R**2-rc**2) )
    end if

    Rad = R
    Vol = V
    disint = 0.

end subroutine init_dais
!------------------------------------------------------------------------------


!------------------------------------------------------------------------------
subroutine dais_step(Ta, SL, Toc, dSL, Rad, Vol, disint)
!  ==========================================================================
! | Calculate current state from previous state
! |
! | Input:
! |   (from timestep = i-1)
! |       Ta:     Antarctic mean surface temperature (degC)
! |       SL:     Sea level (m)
! |       Toc:    High latitude ocean subsurface temperatures [degC]
! |   (from timestep = i)
! |       dSL:    Sea level rate (m/yr)
! |
! | Output:
! |       Rad:    Ice sheet's radius [m]
! |       Vol:    Ice sheet's volume [m3]
! |       disint: Volume of disintegrated ice during this time step [m SLE]
!  ==========================================================================

    implicit none

    real(DP), intent(IN)  :: Ta
    real(DP), intent(IN)  :: SL
    real(DP), intent(IN)  :: Toc
    real(DP), intent(IN)  :: dSL

    real(DP), intent(OUT) :: Rad
    real(DP), intent(OUT) :: Vol
    real(DP), intent(OUT) :: disint

    real(DP) :: hr, rc, P, beta
    real(DP) :: rR, Btot
    real(DP) :: mit, F, ISO
    real(DP) :: Hw, Speed
    real(DP) :: fac
    real(DP) :: c_iso
    real(DP) :: disint_rate

! Start model
    hr   = h0 + c * Ta        ! equation 5
    rc   = (b0 - SL)/slope    ! application of equation 1 (paragraph after eq3)
    P    = P0 * exp(kappa*Ta) ! equation 6
    beta = nu * P**(0.5)      ! equation 7 (corrected with respect to text)

! Total mass accumulation on ice sheet (equation 8)
    if(hr > 0) then
      rR   = R - ((hr - b0 + slope*R)**2) / mu

      Btot = P * Pi * R**2 - &
        Pi * beta * (hr - b0 + slope*R) * (R*R - rR*rR) - &
        (4. * Pi * beta * mu**0.5 *   (R-rR)**2.5) / 5.  + &
        (4. * Pi * beta * mu**0.5 * R*(R-rR)**1.5) / 3.
    else
      Btot = P * Pi*R**2
    end if

! In case there is no marine ice sheet / grounding line
    F   = 0.   ! no ice flux
    ISO = 0.   ! (third term equation 14) NAME?
    fac = Pi * (1.+eps1) * (4./3. * mu**0.5 * R**1.5 - slope*R**2) ! ratio dV/dR (eq 14)

! In case there is a marine ice sheet / grounding line
    if(R > rc) then
      fac   = fac - ( 2.*pi*eps2 * (slope*R**2 - b0*R) ) ! correction fac (eq 14)

      Hw = slope*R - b0 + SL  ! equation 10

  ! Ice speed at grounding line (equation 11)
      Speed = f0 * &
        ((1.-alpha) + alpha * ((Toc - Tf)/(Toc_0 - Tf))**2) * &
        (Hw**gamma) / ( (slope*Rad0 - b0)**(gamma-1.) )
      F     = 2.*Pi*R * del * Hw * Speed   ! equation 9

      ! ISO term depends on dSL_tot (third term equation 14 !! NAME)
      c_iso = 2.*Pi*eps2* (slope*rc**2 - (b0/slope)*rc)

      ! first term is zero if dSL represents only non-AIS components (dSLais=0)
      ! second term is zero if dSL represents all components (dSLais=1)
      ISO =    includes_dSLais  *        c_iso         *  dSL +                        &!dSL = dSL_tot
         (1.-includes_dSLais) * ((1.-c_iso)/c_iso) * (dSL - lf * (Btot - F) / Aoc)  !dSL = dSL_nonAIS
    end if

    ! WAIS/fast dynamics. If temperature is above the trigger temperature and there
    ! is any ice volume left to disintegrate, disintegrate some

!print *, Tcrit, lambda, V, Vmin

!!    if((Ta > Tcrit) .and. (V > 18.0e15)) then
    if(Ta > Tcrit) then
      disint_rate = -lambda*(24.78e15)/57.
    else
      disint_rate = 0.
    end if

    ! Ice sheet volume (equation 13)
    R      = R + tstep*(Btot-F+ISO+disint_rate)/fac
    V      = V + tstep*(Btot-F+ISO+disint_rate)

    Rad = R
    Vol = V
    disint = -disint_rate*tstep*57./24.78e15

end subroutine dais_step
!------------------------------------------------------------------------------

END MODULE dais
