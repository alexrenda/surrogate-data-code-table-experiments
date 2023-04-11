fun (sunPosition, emission) {
  if (sunPosition < 0) {
    ambient = 0;
  } else {
    ambient = sunPosition;
  }
  if (sunPosition < 0.1) {
    emission = emission * 0.1;
  } else {
    emission = emission * sunPosition;
  }
  res = ambient + emission;
  return res;
}
