.. BindCraf2 documentation master file, created by
   sphinx-quickstart on Wed Sep 16 20:47:01 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BindCraf2 documentation
=======================

.. image:: ../.assets/bc2_header.png
   :alt: BindCraft2 header

BindCraft2 (BC2) combines AlphaFold 2, ProteinMPNN, and structural filtering to design protein binders, generate predicted complexes, and rank candidate designs. Its flexible workflow supports the creation of  de novo miniproteins, scaffolded binders, cyclic peptides, antibody formats, and conformational design.

Explore the documentation below to learn how BindCraft2 works, choose the right design approach for your project, and get started with your own designs.

.. raw:: html

   <p>
     <a class="bc2-cta-button" href="design-guide.html" title="Learn how BC2 works, choose a design modality, and understand the workflow end to end">Design Guide</a>
     <a class="bc2-cta-button" href="installation.html" title="Install BC2, check the install, and run campaigns locally, on clusters, or in containers">Installation and Troubleshooting</a>
     <a class="bc2-cta-button" href="reference.html" title="Look up every BC2 setting, its default, and what it controls">Reference Documentation</a>
     <a class="bc2-cta-button" href="outputs.html" title="Understand BC2's output files, measurements, and how to rank and filter designs">Outputs and Measurements</a>
     <a class="bc2-cta-button" href="examples.html" title="Browse worked example campaigns you can run or copy as a starting point">Examples</a>
   </p>
   <style>
     .bc2-cta-button,
     .bc2-cta-button:visited {
       display: inline-block;
       margin: 0.5em 0.5em 1em 0;
       padding: 0.6em 1.4em;
       border-radius: 0.4em;
       background-color: var(--color-brand-primary);
       color: var(--color-background-primary);
       font-weight: 600;
       text-decoration: none;
     }
     .bc2-cta-button:hover,
     .bc2-cta-button:visited:hover {
       background-color: var(--color-brand-content);
       color: var(--color-background-primary);
       text-decoration: none;
     }
   </style>

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   design-guide
   installation
   reference
   outputs
   examples

