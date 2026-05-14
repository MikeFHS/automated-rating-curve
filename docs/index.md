# Automated Rating Curve (ARC)

ARC is a Python tool that generates rating-curve-like hydraulic relationships for each stream cell in a raster domain. Given a DEM, a stream-ID raster, land-cover, and a flow table, ARC:

1. Samples a cross-section for each stream cell
2. Estimates bathymetry (optional)
3. Computes water-surface elevation (WSE), depth, velocity, and top width across discharge increments
4. Writes one or more output datasets (VDT database, curve file, bathymetry raster, etc.)

We recommend reading all documentation in the order it appears in the sidebar.

## Authors and acknowledgment
Mike Follum has been the lead for this project since the code was in it's AutoRoute days. Other contributors include Ahmad Tavakoly, Alan Snow, Joseph Gutenson, Drew Loney, and Ricky Rosas.

Follum Hydrologic Solutions, LLC (FHS) has open-sourced the ARC and Curve2Flood tools to support early flood warning and preparedness in riverine areas lacking adequate flood data or alert systems. These are first-order tools, designed to encourage the development of more advanced modeling systems. We are grateful to all who share in this mission.

The owners of FHS also believe that those with the ability to help others have a responsibility to do so. This belief is rooted in our faith and supported by the following scriptures:

**Hebrews 13:16** - But to do good and to communicate forget not: for with such sacrifices God is well pleased.

**1 Timothy 6:18** - That they do good, that they be rich in good works, ready to distribute, willing to communicate.

**James 4:17** - Therefore to him that knoweth to do good, and doeth it not, to him it is sin.

**Galatians 6:10** - As we have therefore opportunity, let us do good unto all men, especially unto them who are of the household of faith.