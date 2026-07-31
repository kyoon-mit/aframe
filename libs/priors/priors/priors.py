import astropy.cosmology as cosmo
import numpy as np
from bilby.core.prior import (
    ConditionalPowerLaw,
    ConditionalPriorDict,
    ConditionalUniform,
    Constraint,
    Cosine,
    Gaussian,
    LogNormal,
    LogUniform,
    PowerLaw,
    PriorDict,
    Sine,
    Triangular,
    Uniform,
)
from bilby.gw.prior import UniformComovingVolume, UniformSourceFrame
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses

from priors.utils import (
    mass_condition_powerlaw,
    mass_condition_uniform,
    mass_constraints,
)
from utils.cosmology import DEFAULT_COSMOLOGY

# Unit names
msun = r"$M_{\odot}$"
mpc = "Mpc"
rad = "rad"


def uniform_extrinsic() -> PriorDict:
    """
    Define a Bilby `PriorDict` containing distributions that are
    uniform over the allowed ranges of extrinsic binary black hole
    parameters.
    """
    prior = PriorDict()
    prior["dec"] = Cosine()
    prior["ra"] = Uniform(0, 2 * np.pi)
    prior["theta_jn"] = Sine()
    prior["phase"] = Uniform(0, 2 * np.pi)

    return prior


def uniform_spin() -> PriorDict:
    """
    Define a Bilby `PriorDict` containing distributions that are
    uniform over the allowed ranges of binary black hole spin
    parameters.
    """
    prior = PriorDict()
    prior["psi"] = Uniform(0, np.pi)
    prior["a_1"] = Uniform(0, 0.998)
    prior["a_2"] = Uniform(0, 0.998)
    prior["tilt_1"] = Sine(unit=rad)
    prior["tilt_2"] = Sine(unit=rad)
    prior["phi_12"] = Uniform(0, 2 * np.pi)
    prior["phi_jl"] = Uniform(0, 2 * np.pi)
    return prior


def nonspin_bbh(cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY) -> PriorDict:
    """
    Define a Bilby `PriorDict` that describes a reasonable population
    of non-spinning binary black holes

    Masses are defined in the detector frame.

    Args:
        cosmology:
            An `astropy` cosmology, used to determine redshift sampling

    Returns:
        prior:
            `PriorDict` describing the binary black hole population
        detector_frame_prior:
            Boolean indicating which frame masses are defined in
    """
    prior = uniform_extrinsic()
    prior["mass_1"] = Uniform(5, 100, unit=msun)
    prior["mass_2"] = Uniform(5, 100, unit=msun)
    prior["mass_ratio"] = Constraint(0, 1)
    prior["redshift"] = UniformSourceFrame(
        0, 0.5, name="redshift", cosmology=cosmology
    )
    prior["psi"] = 0
    prior["a_1"] = 0
    prior["a_2"] = 0
    prior["tilt_1"] = 0
    prior["tilt_2"] = 0
    prior["phi_12"] = 0
    prior["phi_jl"] = 0

    detector_frame_prior = True
    return prior, detector_frame_prior


def spin_bbh(cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY) -> PriorDict:
    """
    Define a Bilby `PriorDict` that describes a reasonable population
    of spin-aligned binary black holes

    Masses are defined in the detector frame.

    Args:
        cosmology:
            An `astropy` cosmology, used to determine redshift sampling

    Returns:
        prior:
            `PriorDict` describing the binary black hole population
        detector_frame_prior:
            Boolean indicating which frame masses are defined in
    """
    prior = uniform_extrinsic()
    prior["mass_1"] = Uniform(5, 100, unit=msun)
    prior["mass_2"] = Uniform(5, 100, unit=msun)
    prior["mass_ratio"] = Constraint(0, 1)
    prior["redshift"] = UniformSourceFrame(
        0, 0.5, name="redshift", cosmology=cosmology
    )
    prior["psi"] = 0
    prior["a_1"] = Uniform(0, 0.998)
    prior["a_2"] = Uniform(0, 0.998)
    prior["tilt_1"] = Sine(unit=rad)
    prior["tilt_2"] = Sine(unit=rad)
    prior["phi_12"] = 0
    prior["phi_jl"] = 0

    detector_frame_prior = True
    return prior, detector_frame_prior


def end_o3_ratesandpops(
    cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY,
) -> ConditionalPriorDict:
    """
    Define a Bilby `PriorDict` that matches the distributions used
    by the LIGO Rates and Populations group for pipeline searches
    at the end of the third observing run.

    Masses are defined in the source frame.

    Args:
        cosmology:
            An `astropy` cosmology, used to determine redshift sampling

    Returns:
        prior:
            `PriorDict` describing the binary black hole population
        detector_frame_prior:
            Boolean indicating which frame masses are defined in
    """
    prior = ConditionalPriorDict(uniform_extrinsic())
    prior["mass_1"] = PowerLaw(alpha=-2.35, minimum=5, maximum=100, unit=msun)
    prior["mass_2"] = ConditionalPowerLaw(
        condition_func=mass_condition_powerlaw,
        alpha=1,
        minimum=5,
        maximum=100,
        unit=msun,
    )
    prior["redshift"] = UniformComovingVolume(
        0, 2, name="redshift", cosmology=cosmology
    )
    spin_prior = uniform_spin()
    for key, value in spin_prior.items():
        prior[key] = value
    detector_frame_prior = False
    return prior, detector_frame_prior


def end_o3_ratesandpops_bns(
    cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY,
) -> ConditionalPriorDict:
    """
    Define a Bilby `PriorDict` that matches the BNS distribution used
    by the LIGO Rates and Populations group for pipeline searches
    at the end of the third observing run.

    Masses are defined in the source frame.

    Args:
        cosmology:
            An `astropy` cosmology, used to determine redshift sampling

    Returns:
        prior:
            `PriorDict` describing the binary black hole population
        detector_frame_prior:
            Boolean indicating which frame masses are defined in
    """
    prior = ConditionalPriorDict(uniform_extrinsic())
    prior["mass_1"] = Triangular(mode=2.5, minimum=1, maximum=2.5, unit=msun)
    prior["mass_2"] = ConditionalUniform(
        condition_func=mass_condition_uniform,
        minimum=1,
        maximum=2.5,
        unit=msun,
    )
    prior["redshift"] = UniformSourceFrame(
        0, 0.15, name="redshift", cosmology=cosmology
    )
    spin_prior = uniform_spin()
    for key, value in spin_prior.items():
        prior[key] = value
    prior["a_1"] = Uniform(0, 0.4)
    prior["a_2"] = Uniform(0, 0.4)
    detector_frame_prior = False
    return prior, detector_frame_prior


def chirp_mass_to_component_masses(parameters):
    """
    Conversion function to derive component masses
    from chirp mass and mass ratio.
    """
    if "chirp_mass" in parameters and "mass_ratio" in parameters:
        m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(
            parameters["chirp_mass"], parameters["mass_ratio"]
        )
        parameters["mass_1"] = m1
        parameters["mass_2"] = m2
    return parameters


class DetectorFrameFlatChirpPriorDict(PriorDict):
    """Prior whose chirp mass is flat in the DETECTOR frame.

    The pipeline multiplies source-frame masses by (1+z) before generating
    waveforms, so a chirp mass drawn flat in the source frame arrives at the
    model smeared. This sampler works backwards instead: it draws the
    detector-frame chirp mass uniformly, divides by (1+z) to get the
    source-frame value, and only then picks a mass ratio. On rejection
    (component masses outside the box) it redraws the mass ratio alone,
    keeping the chirp mass fixed, so the flat detector-frame marginal is
    exact. The mass-ratio marginal is non-flat by construction.

    The component-mass box is derived from the chirp-mass range itself
    (equal-mass endpoints), which guarantees q = 1 is always acceptable and
    the redraw loop terminates.
    """

    def __init__(self, *, mc_det_min, mc_det_max, z_max, q_min, **kwargs):
        super().__init__(**kwargs)
        self.mc_det_min = mc_det_min
        self.mc_det_max = mc_det_max
        self.q_min = q_min
        # equal-mass endpoints of the source-frame chirp-mass range;
        # chirp_mass(m, m) = m / 2**0.2
        equal_mass_factor = 2**0.2
        self.m_min = (mc_det_min / (1 + z_max)) * equal_mass_factor
        self.m_max = mc_det_max * equal_mass_factor

    def _sample_masses(self, redshift):
        n = len(redshift)
        mc_det = np.random.uniform(self.mc_det_min, self.mc_det_max, n)
        mc_source = mc_det / (1 + redshift)

        q = np.random.uniform(self.q_min, 1.0, n)
        m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(mc_source, q)
        # m1 >= m2 always, so the box check reduces to these two conditions.
        # Redraw only q where it fails: chirp mass stays flat by construction.
        bad = (m1 > self.m_max) | (m2 < self.m_min)
        while bad.any():
            q[bad] = np.random.uniform(self.q_min, 1.0, bad.sum())
            m1_new, m2_new = chirp_mass_and_mass_ratio_to_component_masses(
                mc_source[bad], q[bad]
            )
            m1[bad], m2[bad] = m1_new, m2_new
            bad = (m1 > self.m_max) | (m2 < self.m_min)
        return m1, m2

    def sample(self, size=1, **kwargs):
        samples = super().sample(size=size, **kwargs)
        samples["mass_1"], samples["mass_2"] = self._sample_masses(
            samples["redshift"]
        )
        return samples


def end_o3_ratesandpops_bns_uniform_chirp(
    cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY,
    mc_det_min: float = 0.85,
    mc_det_max: float = 2.75,
    q_min: float = 0.4,
) -> ConditionalPriorDict:
    """
    BNS prior whose chirp mass is uniform in the DETECTOR frame, i.e. flat
    in what the model actually sees after the pipeline applies the (1+z)
    scaling. Spins, redshift and extrinsic parameters match
    `end_o3_ratesandpops_bns`.

    The default range [0.85, 2.75] extends past the analysis region on both
    sides so training edge effects fall outside it. Component masses follow
    from the chirp-mass range (equal-mass endpoints), reaching about
    3.2 solar masses in the source frame near the top edge — intended
    training padding, since chirp mass is the regression target.

    Args:
        cosmology:
            An `astropy` cosmology, used to determine redshift sampling
        mc_det_min:
            Lower edge of the flat detector-frame chirp-mass range
        mc_det_max:
            Upper edge of the flat detector-frame chirp-mass range
        q_min:
            Lower bound of the mass-ratio draw

    Returns:
        prior:
            `PriorDict` describing the binary neutron star population
        detector_frame_prior:
            Boolean indicating which frame masses are defined in
    """
    z_max = 0.15
    prior = DetectorFrameFlatChirpPriorDict(
        mc_det_min=mc_det_min,
        mc_det_max=mc_det_max,
        z_max=z_max,
        q_min=q_min,
        dictionary=uniform_extrinsic(),
    )
    prior["redshift"] = UniformSourceFrame(
        0, z_max, name="redshift", cosmology=cosmology
    )
    spin_prior = uniform_spin()
    for key, value in spin_prior.items():
        prior[key] = value
    prior["a_1"] = Uniform(0, 0.4)
    prior["a_2"] = Uniform(0, 0.4)
    detector_frame_prior = False
    return prior, detector_frame_prior


def gaussian_masses(
    m1: float,
    m2: float,
    sigma: float = 2,
    cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY,
):
    """
    Construct a gaussian bilby prior for masses.

    Masses are defined in the source frame.

    Args:
        m1:
            Mean of the Gaussian distribution for mass 1
        m2:
            Mean of the Gaussian distribution for mass 2
        sigma:
            Standard deviation of the Gaussian distribution for both masses
        cosmology:
            An `astropy` cosmology, used to determine redshift sampling

    Returns:
        prior:
            `PriorDict` describing the binary black hole population
        detector_frame_prior:
            Boolean indicating which frame masses are defined in
    """
    prior = PriorDict(conversion_function=mass_constraints)
    prior["mass_1"] = Gaussian(name="mass_1", mu=m1, sigma=sigma)
    prior["mass_2"] = Gaussian(name="mass_2", mu=m2, sigma=sigma)
    prior["redshift"] = UniformSourceFrame(
        name="redshift", minimum=0, maximum=2, cosmology=cosmology
    )
    prior["dec"] = Cosine(name="dec")
    prior["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )

    detector_frame_prior = False
    return prior, detector_frame_prior


def log_normal_masses(
    m1: float,
    m2: float,
    sigma: float = 2,
    cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY,
):
    """
    Construct a log normal bilby prior for masses.

    Masses are defined in the source frame.

    Args:
        m1:
            Mean of the Log Normal distribution for mass 1
        m2:
            Mean of the Log Normal distribution for mass 2
        sigma:
            Standard deviation for m1 and m2
        cosmology:
            An `astropy` cosmology, used to determine redshift sampling

    Returns:
        prior:
            `PriorDict` describing the binary black hole population
        detector_frame_prior:
            Boolean indicating which frame masses are defined in
    """
    prior = PriorDict(conversion_function=mass_constraints)

    prior["mass_1"] = LogNormal(name="mass_1", mu=np.log(m1), sigma=sigma)
    prior["mass_2"] = LogNormal(name="mass_2", mu=np.log(m2), sigma=sigma)
    prior["mass_ratio"] = Constraint(0.02, 1)

    prior["redshift"] = UniformSourceFrame(
        name="redshift", minimum=0, maximum=2, cosmology=cosmology
    )
    prior["dec"] = Cosine(name="dec")
    prior["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )

    detector_frame_prior = False
    return prior, detector_frame_prior


def ringdown_prior(
    cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY,
) -> (PriorDict, bool):
    """
    Define a Bilby `PriorDict` containing distributions for ringdown parameters

    Quality, Frequency, and Distance are defined in the detector frame

    Args:
        cosmology: An `astropy` cosmology, used to determine distance sampling

    Returns:
        prior: `
            PriorDict` containing the specified distributions
        detector_frame_prior:
            A boolean indicating if the prior is in the detector frame.
    """
    prior = uniform_extrinsic()
    prior["quality"] = Uniform(8, 20)
    prior["frequency"] = LogUniform(100, 1000)
    prior["distance"] = UniformComovingVolume(
        100, 1000, name="luminosity_distance", cosmology=cosmology
    )

    detector_frame_prior = True
    return prior, detector_frame_prior
