import e3nn_jax as e3j
from flax import linen
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tensorial
from tensorial import base


class Particle(tensorial.IrrepsObj):
    """Class for a particle with some attributes"""

    pos = e3j.Irreps("1e")
    mass = tensorial.Attr("0e")
    shielding = tensorial.CartesianTensor("ij=ji", i="1e")


def _particle_value():
    """A complete, valid value dict for the Particle type."""
    shielding = jnp.array([[1.2, 0.0, 1.1], [0.0, 0.0, 1.4], [1.1, 1.4, 3.1]])
    return dict(
        pos=jnp.array([5.2, 5.6, 1.1]),
        mass=12.45,
        shielding=shielding,
    )


def test_basics():
    pos = jnp.array([5.2, 5.6, 1.1])
    mass = 12.45
    shielding = jnp.array([[1.2, 0.0, 1.1], [0.0, 0.0, 1.4], [1.1, 1.4, 3.1]])

    # Make sure we get an error if we don't pass all the attributes
    with pytest.raises(KeyError):
        tensorial.create_tensor(Particle, dict(pos=pos))

    # Test creating a tensor and extracting values
    tensor = tensorial.create_tensor(Particle, dict(pos=pos, mass=mass, shielding=shielding))
    assert jnp.allclose(tensorial.get(Particle, tensor, "pos").array, pos)
    assert jnp.allclose(tensorial.get(Particle, tensor, "mass").array[0].item(), mass)
    # assert jnp.allclose(tensorial.get(Particle, tensor, "shielding"), shielding)

    # Test getting whole tensor
    assert tensorial.get(Particle, tensor) is tensor


def test_cartesian_tensor():
    key = jax.random.PRNGKey(0)
    irreps = e3j.Irreps("1o")
    cart = tensorial.CartesianTensor("ij=ji", i=irreps)

    # Let's create a random Cartesian tensor
    cart_tensor = jax.random.uniform(key, (irreps.dim, irreps.dim))
    cart_tensor = cart_tensor @ cart_tensor.T  # symmetrise

    # Make sure the round trip conversion leads back to the original tensor
    result = cart.from_tensor(cart.create_tensor(cart_tensor))
    assert jnp.allclose(cart_tensor, result)


@pytest.mark.parametrize("rank", [2, 3, 4])
def test_cartesian_tensor_rank(rank):
    key = jax.random.PRNGKey(0)
    irreps = e3j.Irreps("1o")

    # Construct formula: "ij", "ijk", "ijkl"
    indices = "".join([chr(ord("i") + i) for i in range(rank)])
    formula = f"{indices}={indices}"

    # Construct kwargs for CartesianTensor
    kwargs = {idx: irreps for idx in indices}

    cart = tensorial.CartesianTensor(formula, **kwargs)

    # Let's create a random Cartesian tensor
    shape = tuple([irreps.dim] * rank)
    cart_tensor = jax.random.uniform(key, shape)

    # Create tensor
    result = cart.create_tensor(cart_tensor)
    assert isinstance(result, e3j.IrrepsArray)

    # Invert it
    inverted = cart.from_tensor(result)
    assert inverted.shape == cart_tensor.shape


def test_atleast_1d_scalar():
    out = base.atleast_1d(1.5)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (1,)
    assert out[0].item() == pytest.approx(1.5)


def test_atleast_1d_1d_unchanged():
    arr = jnp.arange(3)
    assert base.atleast_1d(arr) is arr


def test_atleast_1d_numpy_backend():
    out = base.atleast_1d(2.0, np_=np)
    assert isinstance(out, np.ndarray)
    assert out.shape == (1,)
    assert out[0] == pytest.approx(2.0)


def test_as_array_irreps_array():
    arr = e3j.IrrepsArray(e3j.Irreps("1o"), jnp.ones(3))
    assert tensorial.as_array(arr) is arr.array


def test_as_array_plain():
    a = jnp.array([1.0, 2.0, 3.0])
    assert tensorial.as_array(a) is a
    assert jnp.allclose(tensorial.as_array([1.0, 2.0]), a[:2])


def test_attr_scalar_round_trip():
    attr = tensorial.Attr("0e")
    tensor = attr.create_tensor(12.5)
    assert isinstance(tensor, e3j.IrrepsArray)
    assert tensor.irreps == e3j.Irreps("0e")
    assert tensor.array.shape == (1,)
    assert tensor.array[0].item() == pytest.approx(12.5)
    # default from_tensor just returns the tensor unchanged
    assert attr.from_tensor(tensor) is tensor


def test_attr_vector_create():
    attr = tensorial.Attr("1o")
    t = attr.create_tensor(jnp.array([1.0, 2.0, 3.0]))
    assert isinstance(t, e3j.IrrepsArray)
    assert t.irreps == e3j.Irreps("1o")


def test_noop():
    irr = e3j.Irreps("1o")
    attr = tensorial.NoOp(irr)
    arr = e3j.IrrepsArray(irr, jnp.eye(3))
    assert attr.create_tensor(arr) is arr
    assert attr.from_tensor(arr) is arr

    wrong_irreps = e3j.IrrepsArray(e3j.Irreps("2e"), jnp.arange(5.0))
    with pytest.raises(AssertionError):
        attr.create_tensor(wrong_irreps)
    with pytest.raises(AssertionError):
        attr.from_tensor(wrong_irreps)


def test_as_irreps():
    attr = tensorial.AsIrreps(e3j.Irreps("2e"))
    t = attr.create_tensor(jnp.arange(5.0))
    assert isinstance(t, e3j.IrrepsArray)
    assert t.irreps == e3j.Irreps("2e")
    assert attr.from_tensor(t) is t

    with pytest.raises(AssertionError):
        attr.create_tensor(jnp.arange(3.0))
    with pytest.raises(AssertionError):
        attr.from_tensor(e3j.IrrepsArray(e3j.Irreps("1o"), jnp.ones(3)))


def test_one_hot():
    attr = tensorial.OneHot(num_classes=3)
    assert attr.num_classes == 3
    assert attr.irreps == e3j.Irreps("3x0e")

    t = attr.create_tensor(jnp.array([[2], [0]]))
    assert t.irreps == e3j.Irreps("3x0e")
    assert jnp.allclose(t.array, np.array([[0, 0, 1], [1, 0, 0]]))

    # Build the type list from explicit types instead
    types = tensorial.OneHot(types=[5, 9])
    assert types.num_classes == 2
    out = types.create_tensor(jnp.array([[9]]))
    assert jnp.allclose(out.array, np.array([[0, 1]]))

    with pytest.raises(ValueError):
        tensorial.OneHot()


def test_spherical_harmonic():
    sh = tensorial.SphericalHarmonic("1o", normalise=True)
    vec = jnp.array([1.0, 0.0, 0.0])

    # plain-array input exercises the as_array fallback branch
    out = sh.create_tensor(vec)
    assert isinstance(out, e3j.IrrepsArray)
    assert out.irreps == e3j.Irreps("1o")

    # IrrepsArray input should give the same result
    out_irrep = sh.create_tensor(e3j.IrrepsArray(e3j.Irreps("1o"), vec))
    assert jnp.allclose(out.array, out_irrep.array)


def test_create_particle():
    value = _particle_value()
    out = tensorial.create(Particle, value)
    assert set(out) == {"pos", "mass", "shielding"}
    assert isinstance(out["pos"], e3j.IrrepsArray)
    assert jnp.allclose(out["pos"].array, value["pos"])
    assert out["mass"].array[0].item() == pytest.approx(12.45)
    assert isinstance(out["shielding"], e3j.IrrepsArray)

    # missing attribute should raise
    with pytest.raises(KeyError):
        tensorial.create(Particle, dict(pos=value["pos"]))


def test_create_irrepsobj_instance():
    # the instance branch treats the object as a single leaf -> one concatenated tensor
    instance = Particle()
    out = tensorial.create(instance, _particle_value())
    assert isinstance(out, e3j.IrrepsArray)
    expected = tensorial.create(Particle, _particle_value())
    assert jnp.allclose(out.array, e3j.concatenate(list(expected.values())).array)


def test_irreps_particle():
    irr = tensorial.irreps(Particle)
    expected = e3j.Irreps("1e") + e3j.Irreps("0e") + Particle.shielding.irreps
    assert irr == expected
    assert irr.dim == Particle.pos.dim + 1 + Particle.shielding.irreps.dim


def test_irreps_leaves():
    attr = tensorial.Attr("0e")
    assert tensorial.irreps(attr) == e3j.Irreps("0e")

    irr_in = e3j.Irreps("2e")
    assert tensorial.irreps(irr_in) is irr_in


def test_create_tensor_dict():
    spec = {"a": e3j.Irreps("1o"), "b": tensorial.Attr("2e")}
    a = jnp.arange(3.0)
    b = jnp.arange(5.0)
    t = tensorial.create_tensor(spec, {"a": a, "b": b})
    assert t.irreps == e3j.Irreps("1o + 2e")
    assert jnp.allclose(t.array[:3], a)
    assert jnp.allclose(t.array[3:], b)


def test_create_tensor_frozen_dict():
    spec = linen.FrozenDict({"a": e3j.Irreps("1o"), "b": "2e"})
    t = tensorial.create_tensor(spec, {"a": jnp.arange(3.0), "b": jnp.arange(5.0)})
    assert t.irreps == e3j.Irreps("1o + 2e")


def test_create_tensor_irreps_and_str():
    t1 = tensorial.create_tensor("1o", jnp.ones(3))
    t2 = tensorial.create_tensor(e3j.Irreps("1o"), jnp.ones(3))
    assert t1.irreps == e3j.Irreps("1o")
    assert t2.irreps == t1.irreps
    assert jnp.allclose(t1.array, t2.array)


def test_create_tensor_irrepsobj_instance():
    instance = Particle()
    t = tensorial.create_tensor(instance, _particle_value())
    assert isinstance(t, e3j.IrrepsArray)
    assert t.irreps == tensorial.irreps(Particle)


def test_create_tensor_unrecognised():
    with pytest.raises(TypeError):
        tensorial.create_tensor(42, jnp.ones(3))


def test_from_tensor_particle_round_trip():
    value = _particle_value()
    tensor = tensorial.create_tensor(Particle, value)
    parts = tensorial.from_tensor(Particle, tensor)
    assert set(parts) == {"pos", "mass", "shielding"}
    assert jnp.allclose(parts["pos"].array, value["pos"])
    assert parts["mass"].array[0].item() == pytest.approx(value["mass"])
    # CartesianTensor.from_tensor inverts the change of basis back to Cartesian
    assert jnp.allclose(parts["shielding"], value["shielding"])


def test_from_tensor_instance():
    tensor = tensorial.create_tensor(Particle(), _particle_value())
    parts = tensorial.from_tensor(Particle(), tensor)
    assert set(parts) == {"pos", "mass", "shielding"}


def test_from_tensor_frozen_dict():
    spec = linen.FrozenDict({"a": e3j.Irreps("1o"), "b": e3j.Irreps("0e")})
    a, b = jnp.arange(3.0), jnp.array([4.0])
    tensor = tensorial.create_tensor(spec, {"a": a, "b": b})
    parts = tensorial.from_tensor(spec, tensor)
    assert jnp.allclose(parts["a"].array, a)
    assert jnp.allclose(parts["b"].array, b)


def test_from_tensor_irreps_mismatch():
    bad = e3j.IrrepsArray(e3j.Irreps("2e"), jnp.ones(5))
    with pytest.raises(ValueError):
        tensorial.from_tensor(e3j.Irreps("1e"), bad)


def test_from_tensor_irreps_ok():
    arr = e3j.IrrepsArray(e3j.Irreps("1o"), jnp.eye(3))
    assert tensorial.from_tensor(e3j.Irreps("1o"), arr) is arr


def test_from_tensor_unrecognised():
    with pytest.raises(TypeError):
        tensorial.from_tensor(42, jnp.ones(3))


def test_tensorial_attrs_class():
    attrs = tensorial.tensorial_attrs(Particle)
    assert list(attrs) == ["pos", "mass", "shielding"]
    assert attrs["pos"] == e3j.Irreps("1e")
    assert attrs["mass"] == Particle.mass
    assert attrs["shielding"] == Particle.shielding


def test_tensorial_attrs_instance():
    p = Particle()
    p.extra = e3j.Irreps("1o")  # noqa: SLF001 - instance attr on purpose
    attrs = tensorial.tensorial_attrs(p)
    assert set(attrs) == {"pos", "mass", "shielding", "extra"}
    assert attrs["extra"] == e3j.Irreps("1o")


def test_tensorial_attrs_dict():
    d = {"a": e3j.Irreps("1o"), "_private": e3j.Irreps("0e")}
    assert list(tensorial.tensorial_attrs(d)) == ["a"]


def test_tensorial_attrs_frozen_dict():
    fd = linen.FrozenDict({"a": e3j.Irreps("1o"), "_b": e3j.Irreps("0e")})
    assert list(tensorial.tensorial_attrs(fd)) == ["a"]


def test_tensorial_attrs_bad_type():
    with pytest.raises(TypeError):
        tensorial.tensorial_attrs(42)


def test_get_slices():
    tensor = tensorial.create_tensor(Particle, _particle_value())
    pos_dim = Particle.pos.dim
    mass_dim = 1

    # each named slice must match the corresponding piece of the flat array
    assert tensorial.get(Particle, tensor, "pos").shape == (pos_dim,)
    assert tensorial.get(Particle, tensor, "mass").shape == (mass_dim,)
    assert tensorial.get(Particle, tensor, "shielding").irreps == Particle.shielding.irreps

    # empty/None attr name returns the tensor unchanged
    assert tensorial.get(Particle, tensor, "") is tensor


def test_get_unknown_attr():
    with pytest.raises(ValueError):
        tensorial.get(Particle, jnp.arange(10.0), "nope")


def test_create_non_irrepsobj_raises():
    # create() default branch raises TypeError for a non-IrrepsObj class
    class NotTensorial:
        x = e3j.Irreps("1o")

    with pytest.raises(TypeError):
        tensorial.create(NotTensorial, {"x": jnp.ones(3)})


def test_irreps_non_irrepsobj_raises():
    class NotTensorial:
        x = e3j.Irreps("1o")

    with pytest.raises(TypeError):
        tensorial.irreps(NotTensorial)


def test_irreps_invalid_attr_raises():
    # an attribute that is tensorial-looking but has no resolvable irreps
    class Broken(tensorial.IrrepsObj):
        bad = 42

    with pytest.raises(AttributeError):
        tensorial.irreps(Broken)


def test_tensorial_attrs_non_irrepsobj_class_raises():
    class NotTensorial:
        x = 1

    with pytest.raises(TypeError):
        tensorial.tensorial_attrs(NotTensorial)
