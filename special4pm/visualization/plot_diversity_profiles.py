from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [5, 4]
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

plt.plot([36, 34.2, 33.0], label="Equal", linestyle="--")
plt.plot([36, 27.6, 22.4], label="Added", linestyle="--")
plt.plot([25, 18.4, 14.7], label="Base", linestyle="--")
plt.plot([16, 7.7, 4.7], label="Removed", linestyle="--")

plt.xticks([0, 1, 2],[0,1,2])


#ax1.set_title(names[0], fontsize=24)
#ax1.set_xlabel("Species Rank", fontsize=20)
#ax1.set_ylabel("Occurences", fontsize=20)
#ax1.set_xticks([0, no_species - 1])
#ax1.set_xticklabels([1, no_species])
#plt.yticks([0, max(estimations[0].reference_sample_abundance.values()) - 1],
#           [0, max(estimations[0].reference_sample_abundance.values()) - 1])
plt.xlabel("Order q", fontsize=20)
plt.ylabel("Hill Number", fontsize=20)

plt.legend()
plt.tight_layout()
plt.savefig("figures/drift_eval_diversity_profiles.pdf", format="pdf")
#plt.show()
plt.close()
# plt.show()

plt.rcParams['figure.figsize'] = [5, 4]
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

plt.plot([717, 45.8, 22.5], label="BPI-2018", linestyle="--")
plt.plot([618, 30.4, 18.7], label="BPI-2019", linestyle="--")
plt.plot([164, 39.6, 23.9], label="Sepsis", linestyle="--")
plt.plot([150, 36.9, 18.8], label="BPI-2012", linestyle="--")

plt.xticks([0,1,2],[0,1,2])
plt.yscale("log")

#ax1.set_title(names[0], fontsize=24)
#ax1.set_xlabel("Species Rank", fontsize=20)
#ax1.set_ylabel("Occurences", fontsize=20)
#ax1.set_xticks([0, no_species - 1])
#ax1.set_xticklabels([1, no_species])
#plt.yticks([0, max(estimations[0].reference_sample_abundance.values()) - 1],
#           [0, max(estimations[0].reference_sample_abundance.values()) - 1])
plt.xlabel("Order q", fontsize=20)
plt.ylabel("Hill Number", fontsize=20)
plt.ylim(0,1000)
plt.legend()
plt.tight_layout()
plt.savefig("figures/real_eval_diversity_profiles.pdf", format="pdf")
#plt.show()
plt.close()
# plt.show()