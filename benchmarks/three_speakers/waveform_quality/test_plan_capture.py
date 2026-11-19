"""Check the capture planner against known source and phase cases."""

import unittest

import numpy as np

from plan_capture import estimate, isolation_needed


class CapturePlanTest(unittest.TestCase):
    def test_isolated_capture_needs_no_reduction(self):
        t=np.arange(16000)/16000
        takes=[]
        for source, pitch in enumerate((211,293,367)):
            x=np.zeros((len(t),3))
            x[:,source]=.05*np.sin(2*np.pi*pitch*t)
            takes.append(x)
        report=estimate(takes,16000,quiet_levels=(20,))
        for target in ("96","98"):
            self.assertEqual(set(report["anti_leakage_db_by_mic"][target].values()),{0})
            replay=report["simulation_hypothetical_bleed_reduction"][target]
            self.assertEqual(replay["trials_passing"],4)

    def test_required_dbs_exact_for_uncorrelated_leak(self):
        t=np.arange(16000)/16000
        x=np.sin(2*np.pi*211*t)
        a=np.sin(2*np.pi*293*t)
        b=np.sin(2*np.pi*367*t)
        leak=np.stack([.2*a,.2*b])
        observed,bound=isolation_needed(x,leak,.98)
        expected=10*np.log10((.2**2+.2**2)/(1/.98-1))
        self.assertAlmostEqual(observed,expected,places=9)
        self.assertGreater(bound,observed)
        restored=x+10**(-observed/20)*leak.sum(axis=0)
        self.assertAlmostEqual(np.corrcoef(x,restored)[0,1]**2,.98,places=9)

    def test_phase_margin_protects_simultaneous_coherent_leak(self):
        t=np.arange(16000)/16000
        target=np.sin(2*np.pi*211*t)
        transformed=np.sin(2*np.pi*293*t)
        exact,bound=isolation_needed(target,np.stack([.3*transformed,-.3*transformed]),.96)
        self.assertEqual(exact,0)
        self.assertGreater(bound,8)

if __name__=="__main__":
    unittest.main()
