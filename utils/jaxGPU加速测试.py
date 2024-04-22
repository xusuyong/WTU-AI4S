import jax  
print(jax.lib.xla_bridge.get_backend().platform)
