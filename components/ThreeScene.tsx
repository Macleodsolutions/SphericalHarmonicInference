import React, {useEffect, useRef, useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import editShaderForSHC from "../utils/shaderHelper";
import {MeshBasicMaterial, WebGLRenderer} from "three";

interface ThreeSceneProps {
    width: number;
    height: number;
    imageSrc?: string;
    label?: [];
}

const scene = new THREE.Scene();
const skySphereGeometry = new THREE.SphereGeometry(500, 60, 40).scale(-1, 1, 1);
const sphereGeometry = new THREE.SphereGeometry(1, 32, 32);
const textureLoader = new THREE.TextureLoader();
let renderer: WebGLRenderer;

const ThreeScene: React.FC<ThreeSceneProps> = ({ width, height, imageSrc, label }) => {
    //Scene setup
    const containerRef = useRef<HTMLDivElement>(null);
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 5;
    const [sphereMaterial] = useState<THREE.MeshBasicMaterial>(new THREE.MeshBasicMaterial({ color: 0xffffff }));
    const sphereMaterialRef = useRef(sphereMaterial);
    const skySphere = new THREE.Mesh(skySphereGeometry, new MeshBasicMaterial());
    skySphere.rotateY(1.57);
    scene.add(skySphere);
    //Add onBeforeCompile
    editShaderForSHC(sphereMaterial);
    useEffect(() => {
        if(!containerRef.current){return;}
        renderer = new THREE.WebGLRenderer();
        containerRef.current.appendChild(renderer.domElement);
        const controls = new OrbitControls(camera, renderer.domElement);

        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterialRef.current);
        scene.add(sphere);

        sphereMaterialRef.current.userData.SphericalHarmonicCoefficients = {value: []};

        const animate = () => {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        };

        animate();

        return () => {
            containerRef.current?.removeChild(renderer.domElement);
            controls.dispose();
            renderer.dispose();
            sphereMaterialRef.current.dispose();
            skySphereGeometry.dispose();
            sphereGeometry.dispose();
        };
    }, []);
    useEffect(() => {
        if (!containerRef.current || !renderer) return;
        renderer.setSize(width, height);
    }, [width, height]);

    useEffect(() => {
        if (!label || !sphereMaterialRef.current || !imageSrc) return;
        sphereMaterialRef.current.userData.SphericalHarmonicCoefficients.value = new Float32Array(label);
        skySphere.material.map = textureLoader.load(imageSrc);
    }, [label, imageSrc]);

    return <div ref={containerRef} />;
};

export default ThreeScene;
