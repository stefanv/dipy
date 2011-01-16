""" Test fsl package """

import numpy as np

import nibabel as nib

from .. import fsl
from ...data import get_data

from nose.tools import assert_true, assert_false, assert_equal, assert_raises

from numpy.testing.decorators import skipif
from numpy.testing import assert_array_equal, assert_array_almost_equal

flirt_test = skipif(not fsl.have_flirt())

_small_img = nib.load(get_data()[0])
ONE_VOL = nib.Nifti1Image(_small_img.get_data()[...,0],
                         _small_img.get_affine(),
                         _small_img.get_header())


@flirt_test
def test_flirt_behavior():
    # self to self is identity
    res0 = fsl.flirt(ONE_VOL, ONE_VOL)
    assert_array_almost_equal(res0, np.eye(4))
    # pos determinant version of same image
    pos_det = nib.as_closest_canonical(ONE_VOL)
    # original to pos det version has x flip
    res1 = fsl.flirt(ONE_VOL, pos_det)
    flipx = np.diag([-1,1,1,1])
    flipx[0,3] = ONE_VOL.get_shape()[0]-1
    assert_array_almost_equal(res1, flipx)
    # Either way round
    res2 = fsl.flirt(pos_det, ONE_VOL)
    assert_array_almost_equal(res2, flipx)
    # pos det to self obviously also identity
    res3 = fsl.flirt(pos_det, pos_det)
    assert_array_almost_equal(res3, np.eye(4))
    # Mapping is from ``in`` to ``ref`` voxels
    arr = ONE_VOL.get_data()
    arr = arr[:,:,3:]
    anat_down = nib.Nifti1Image(arr,
                                ONE_VOL.get_affine(),
                                ONE_VOL.get_header())
    res4 = fsl.flirt(anat_down, ONE_VOL)
    assert_array_almost_equal(res4, [[1,0,0,0],
                                     [0,1,0,0],
                                     [0,0,1,3],
                                     [0,0,0,1]])

