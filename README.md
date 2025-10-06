# Neutrino-Physics-Thesis
Simulation of Neutrino oscillation in a Collapsar, developed as part of my Master's Thesis.

The thesis focused on understanding neutrino physics and simulating the evolution of a neutrino that is produced at the centre of the collapsing star and seeing how the flavor composition evolves as a function of distance, angle and time from the collapsar simulation. 
The data comes from a 3D magnetorotational core collapse simulation made by M.Obergaulinger and M.Á. Aloy. The progenitor is a 13M superluminous supernova that ends as a black hole. (See [https://arxiv.org/pdf/2008.07205] for the 3D setup and [https://arxiv.org/pdf/2108.13864] for the 13M superluminous progenitor). 
(Note: The fifth paper on this specific progenitor and results still hasn't been published; when done, the ref will be added.) 

The study of neutrino evolution was done in the following order, where a code has been developed to solve the problem.
1. Analyse the data and proceed to an interpolation scheme of the matter density and the electron fraction over the distance, to all angles and times available. Compute the resonant layers and plot them on the matter density profile. 
2. The adiabaticity parameter and flip probability are computed for a set of data to probe if any flip in the mass eigenstate is triggered at the resonant layers.
3. Neutrino Flavor Evolution for 2 flavors. A simple way to see where the MSW effect is triggered. Initially, the study started directly with 3 flavors, but the results did not work for an initial muon or tau neutrino. Therefore, this code was mainly to see if the initial code was working whilst using data. 
4. Neutrino Flavor Evolution for 3 flavors. Complete evolution of a neutrino being produced at the centre of the collapsar and evolving towards the outer region and escaping. This code can actually be used for any data of matter density profile. It needs the distance in (cm), the matter density in (g/cc) and the electron fraction in (Ø). The code works for normal and inverted hierarchy, all 6 flavors (including anti), the value of delta CP phase and different energies.
5. Maximal and minimal value of the CP phase. This code allows us to see the possible values that the muon and tau neutrino flavors can take as a function of the delta CP phase. 
6. Neutrino Flavor Evolution for 4 flavors. This part was a bonus for the thesis and added a sterile neutrino in the simulation. The values of the different angles were found from different papers. Results may work for an initial muon and tau neutrino, but show inconsistent results for an initial electron neutrino (due to the value of the mixing angles).
(Note: For this 6th part, it might be possible to constrain the values of the mixing angles by assuming that the final probability of a sterile neutrino should be below a certain threshold. 
