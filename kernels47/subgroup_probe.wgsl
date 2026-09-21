enable subgroups;

@group(0) @binding(0) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size) sg_size: u32,
  @builtin(subgroup_id) sg: u32,
  @builtin(subgroup_invocation_id) lane: u32,
) {
  if (lid.x == 0u) {
    out[0] = sg_size;
    out[1] = sg;
    out[2] = lane;
  }
}
