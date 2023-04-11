fun (sunPosition, emission) {
  ambient = sunPosition;
  emission = emission * sunPosition;
  res = ambient + emission;
  return res;
}
