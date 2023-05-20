import { MeshBasicMaterial } from "three";

function editShaderForSHC(material: MeshBasicMaterial) {
  // Initialize SHC in the material's user data
  material.userData.SphericalHarmonicCoefficients = { value: [] };

  material.onBeforeCompile = (shader) => {
    shader.uniforms.SphericalHarmonicCoefficients =
      material.userData.SphericalHarmonicCoefficients;

    shader.vertexShader = `
                varying vec3 vNormal;
                ${shader.vertexShader}
            `.replace(
      "#include <project_vertex>",
      `
                #include <project_vertex>
                vNormal = normal;
            `
    );

    // Modify the fragment shader to use SHC for calculating light irradiance
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <clipping_planes_pars_fragment>",
      `
            // Define a structure to hold the SHC
            struct SHCoefficients {
                    vec3 l00, l1m1, l10, l11, l2m2, l2m1, l20, l21, l22;
                };
                
            // Define SHC as uniform array of 27 floats
            uniform float SphericalHarmonicCoefficients[27];
            
            // Function to calculate the irradiance from the normal vector and SHC
            vec3 calcIrradiance(vec3 nor, SHCoefficients c) {
                // The constants used in the calculation
                const float c1 = 0.429043;
                const float c2 = 0.511664;
                const float c3 = 0.743125;
                const float c4 = 0.886227;
                const float c5 = 0.247708;
                
                // The calculation itself
                return (
                    c1 * c.l22 * (nor.x * nor.x - nor.y * nor.y) +
                    c3 * c.l20 * nor.z * nor.z +
                    c4 * c.l00 -
                    c5 * c.l20 +
                    2.0 * c1 * c.l2m2 * nor.x * nor.y +
                    2.0 * c1 * c.l21  * nor.x * nor.z +
                    2.0 * c1 * c.l2m1 * nor.y * nor.z +
                    2.0 * c2 * c.l11  * nor.x +
                    2.0 * c2 * c.l1m1 * nor.y +
                    2.0 * c2 * c.l10  * nor.z
                );
            }
            
            // Function to create an SHCoefficients structure from a float array
            SHCoefficients createSHCoefficients(float data[27]) {
                SHCoefficients c;
                c.l00 = vec3(data[0], data[1], data[2]);
                c.l1m1 = vec3(data[3], data[4], data[5]);
                c.l10 = vec3(data[6], data[7], data[8]);
                c.l11 = vec3(data[9], data[10], data[11]);
                c.l2m2 = vec3(data[12], data[13], data[14]);
                c.l2m1 = vec3(data[15], data[16], data[17]);
                c.l20 = vec3(data[18], data[19], data[20]);
                c.l21 = vec3(data[21], data[22], data[23]);
                c.l22 = vec3(data[24], data[25], data[26]);
                return c;
            }
            
            #include <clipping_planes_pars_fragment>
            `
    );

    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <dithering_fragment>",
      `
            // Create SHCoefficients structure from the uniform array
            SHCoefficients shCoefficients = createSHCoefficients(SphericalHarmonicCoefficients);
            
            // Calculate the irradiance from the normal vector and the SHC
            vec3 irradiance = calcIrradiance(vNormal, shCoefficients);
            
            // Multiply the color by the irradiance
            gl_FragColor.rgb *= irradiance;
            
            // Include the original shader code
            #include <dithering_fragment>
            `
    );
  };

  return material;
}

export default editShaderForSHC;
