import unittest
import ezgal.cosmology
import tests
import numpy as np


class test_lengths(unittest.TestCase):

    cosmo = None

    def setUp(self):

        # standard cosmology
        self.cosmo = ezgal.cosmology.Cosmology(
            Om=0.272, Ol=0.728, h=0.704, w=-1)

    def test_Otot(self):

        # 1 because the universe is flat!
        self.assertAlmostEqual(self.cosmo.Otot(), 1, 7)

    def test_Ok(self):

        # 0, also because the universe is flat!
        self.assertAlmostEqual(self.cosmo.Ok(), 0, 7)

    def test_h0(self):

        # hubble constant: such a classic
        self.assertAlmostEqual(self.cosmo.H0(), 70.40, 2)

    # test a few points between efunc and tfunc
    # tfunc calls efunc and does some simple
    # arithmetic.  Both are used extensively by
    # everything else.
    def test_efunc_0(self):

        self.assertAlmostEqual(self.cosmo.Efunc(0), 1, 7)

    def test_efunc_30(self):

        self.assertAlmostEqual(self.cosmo.Efunc(3.0), 0.2348168, 7)

    def test_tfunc_05(self):

        self.assertAlmostEqual(self.cosmo.Tfunc(0.5), 0.5196295, 7)

    def test_tfunc_50(self):

        self.assertAlmostEqual(self.cosmo.Tfunc(5.0), 0.0216104, 7)

    def test_Hz(self):

        self.assertAlmostEqual(self.cosmo.Hz(1), 119.9695321, 7)

    def test_a(self):

        self.assertAlmostEqual(self.cosmo.a(1), 0.5, 2)

    def test_Dh(self):

        self.assertAlmostEqual(self.cosmo.Dh() / 1e26, 1.3139987, 7)

    def test_Dh_cm(self):

        self.assertAlmostEqual(self.cosmo.Dh(cm=True) / 1e28, 1.3139987, 7)

    def test_Dh_meters(self):

        self.assertAlmostEqual(self.cosmo.Dh(meter=True) / 1e26, 1.3139987, 7)

    def test_Dh_pc(self):

        self.assertAlmostEqual(self.cosmo.Dh(pc=True) / 1e9, 4.25838068, 7)

    def test_Dh_kpc(self):

        self.assertAlmostEqual(self.cosmo.Dh(kpc=True) / 1e6, 4.25838068, 7)

    def test_Dh_mpc(self):

        self.assertAlmostEqual(self.cosmo.Dh(mpc=True) / 1e3, 4.25838068, 7)

    def test_Dc(self):

        self.assertAlmostEqual(self.cosmo.Dc(1) / 1e26, 1.03107037, 7)

    def test_Dc_cm(self):

        self.assertAlmostEqual(self.cosmo.Dc(1, cm=True) / 1e28, 1.03107037, 7)

    def test_Dc_meters(self):

        self.assertAlmostEqual(
            self.cosmo.Dc(1, meter=True) / 1e26,
            1.03107037, 7)

    def test_Dc_pc(self):

        self.assertAlmostEqual(self.cosmo.Dc(1, pc=True) / 1e9, 3.3414721, 7)

    def test_Dc_kpc(self):

        self.assertAlmostEqual(self.cosmo.Dc(1, kpc=True) / 1e6, 3.3414721, 7)

    def test_Dc_mpc(self):

        self.assertAlmostEqual(self.cosmo.Dc(1, mpc=True) / 1e3, 3.3414721, 7)

    def test_Dm(self):

        self.assertAlmostEqual(self.cosmo.Dm(1) / 1e26, 1.03107037, 7)

    def test_Dm_cm(self):

        self.assertAlmostEqual(self.cosmo.Dm(1, cm=True) / 1e28, 1.03107037, 7)

    def test_Dm_meters(self):

        self.assertAlmostEqual(
            self.cosmo.Dm(1, meter=True) / 1e26,
            1.03107037, 7)

    def test_Dm_pc(self):

        self.assertAlmostEqual(self.cosmo.Dm(1, pc=True) / 1e9, 3.3414721, 7)

    def test_Dm_kpc(self):

        self.assertAlmostEqual(self.cosmo.Dm(1, kpc=True) / 1e6, 3.3414721, 7)

    def test_Dm_mpc(self):

        self.assertAlmostEqual(self.cosmo.Dm(1, mpc=True) / 1e3, 3.3414721, 7)

    def test_Da(self):

        self.assertAlmostEqual(self.cosmo.Da(1) / 1e25, 5.1553519, 7)

    def test_Da_cm(self):

        self.assertAlmostEqual(self.cosmo.Da(1, cm=True) / 1e27, 5.1553519, 7)

    def test_Da_meters(self):

        self.assertAlmostEqual(
            self.cosmo.Da(1, meter=True) / 1e25,
            5.1553519, 7)

    def test_Da_pc(self):

        self.assertAlmostEqual(self.cosmo.Da(1, pc=True) / 1e9, 1.6707361, 7)

    def test_Da_kpc(self):

        self.assertAlmostEqual(self.cosmo.Da(1, kpc=True) / 1e6, 1.6707361, 7)

    def test_Da_mpc(self):

        self.assertAlmostEqual(self.cosmo.Da(1, mpc=True) / 1e3, 1.6707361, 7)

    def test_Da2(self):

        self.assertAlmostEqual(self.cosmo.Da2(1, 1.5) / 1e25, 1.3490114, 7)

    def test_Da2_cm(self):

        self.assertAlmostEqual(
            self.cosmo.Da2(1, 1.5, cm=True) / 1e27,
            1.3490114, 7)

    def test_Da2_meters(self):

        self.assertAlmostEqual(
            self.cosmo.Da2(1, 1.5, meter=True) / 1e25,
            1.3490114, 7)

    def test_Da2_pc(self):

        self.assertAlmostEqual(
            self.cosmo.Da2(1, 1.5, pc=True) / 1e8,
            4.3718491, 7)

    def test_Da2_kpc(self):

        self.assertAlmostEqual(
            self.cosmo.Da2(1, 1.5, kpc=True) / 1e5,
            4.3718491, 7)

    def test_Da2_mpc(self):

        self.assertAlmostEqual(
            self.cosmo.Da2(1, 1.5, mpc=True) / 1e2,
            4.3718491, 7)

    def test_Dl(self):

        self.assertAlmostEqual(self.cosmo.Dl(1) / 1e26, 2.0621407, 7)

    def test_Dl_cm(self):

        self.assertAlmostEqual(self.cosmo.Dl(1, cm=True) / 1e28, 2.0621407, 7)

    def test_Dl_meters(self):

        self.assertAlmostEqual(
            self.cosmo.Dl(1, meter=True) / 1e26,
            2.0621407, 7)

    def test_Dl_pc(self):

        self.assertAlmostEqual(self.cosmo.Dl(1, pc=True) / 1e9, 6.6829443, 7)

    def test_Dl_kpc(self):

        self.assertAlmostEqual(self.cosmo.Dl(1, kpc=True) / 1e6, 6.6829443, 7)

    def test_Dl_mpc(self):

        self.assertAlmostEqual(self.cosmo.Dl(1, mpc=True) / 1e3, 6.6829443, 7)

    def test_DistMod(self):

        # for any non-astronomy peeps: the distance modulus is in units of
        # magnitudes, not length (despite the name), so no need to test
        # unit conversion
        self.assertAlmostEqual(self.cosmo.DistMod(1), 44.1248392, 7)

    def test_scale(self):

        self.assertAlmostEqual(self.cosmo.scale(1) / 1e20, 2.4993851, 7)

    def test_scale_cm(self):

        self.assertAlmostEqual(
            self.cosmo.scale(1, cm=True) / 1e22,
            2.4993851, 7)

    def test_scale_meters(self):

        self.assertAlmostEqual(
            self.cosmo.scale(1, meter=True) / 1e20,
            2.4993851, 7)

    def test_scale_pc(self):

        self.assertAlmostEqual(
            self.cosmo.scale(1, pc=True) / 1e3,
            8.0999570, 7)

    def test_scale_kpc(self):

        self.assertAlmostEqual(self.cosmo.scale(1, kpc=True), 8.0999570, 7)

    def test_scale_mpc(self):

        self.assertAlmostEqual(
            self.cosmo.scale(1, mpc=True) * 1e3,
            8.0999570, 7)

    def test_lengthConversion_cm(self):

        self.assertAlmostEqual(self.cosmo.lengthConversion(cm=True), 100, 0)

    def test_lengthConversion_meters(self):

        self.assertAlmostEqual(self.cosmo.lengthConversion(meter=True), 1.0, 0)

    def test_lengthConversion_pc(self):

        self.assertAlmostEqual(
            self.cosmo.lengthConversion(pc=True) * 1e17,
            3.2407799,
            7)

    def test_lengthConversion_kpc(self):

        self.assertAlmostEqual(
            self.cosmo.lengthConversion(kpc=True) * 1e20,
            3.2407799,
            7)

    def test_lengthConversion_mpc(self):

        self.assertAlmostEqual(
            self.cosmo.lengthConversion(mpc=True) * 1e23,
            3.2407799,
            7)


if __name__ == '__main__':
    unittest.main()
