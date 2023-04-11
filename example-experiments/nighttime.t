fun (sunPosition, emission) {
  ambient = 0.0;
  emission = emission * 0.1;
  res = ambient + emission;
  return res;
}
