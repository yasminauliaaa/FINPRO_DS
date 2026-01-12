import streamlit as st

def contact_me():
    # Header dengan styling
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #667eea; font-size: 3rem;">📬 Contact Me</h1>
            <p style="font-size: 1.2rem; color: #888; margin-top: 1rem;">
                Let's connect and collaborate on exciting data science projects!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Profile section
    col1, col2 = st.columns([3, 7])
    
    with col1:
        # Bisa tambahkan foto profil
        st.markdown("""
            <div style="text-align: center;">
                <div style="
                    width: 150px; 
                    height: 150px; 
                    border-radius: 50%; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto;
                    font-size: 4rem;
                ">
                    👩‍💻
                </div>
                <h3 style="margin-top: 1rem;">Yasmin Aulia</h3>
                <p style="color: #888;">Data Analyst & Science Enthusiast</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="padding: 1rem;">
                <h3 style="color: #667eea; margin-bottom: 1.5rem;">🤝 Mari Terhubung!</h3>
                <p style="font-size: 1.1rem; line-height: 1.8; text-align: justify;">
                    Saya terbuka untuk kolaborasi, diskusi, dan berbagi pengetahuan seputar 
                    <b>Data Analyst</b>, <b>Data Science</b>, <b>Machine Learning</b>, dan <b>Business Analytics</b>. 
                    Jangan ragu untuk menghubungi saya melalui channel di bawah ini!
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Contact cards
    st.markdown("### 📱 Get in Touch")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.3s;
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📧</div>
                <h3 style="color: white; margin-bottom: 0.5rem;">Email</h3>
                <a href="mailto:yasminaulia715@gmail.com" style="
                    color: white; 
                    text-decoration: none; 
                    font-weight: 600;
                    word-break: break-all;
                ">
                    yasminaulia715@gmail.com
                </a>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #0077B5 0%, #00A0DC 100%);
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">💼</div>
                <h3 style="color: white; margin-bottom: 0.5rem;">LinkedIn</h3>
                <a href="https://www.linkedin.com/in/yasminauliaa/" target="_blank" style="
                    color: white; 
                    text-decoration: none; 
                    font-weight: 600;
                ">
                    /yasminauliaa
                </a>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #333 0%, #555 100%);
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🐱</div>
                <h3 style="color: white; margin-bottom: 0.5rem;">GitHub</h3>
                <a href="https://github.com/yasminauliaaa" target="_blank" style="
                    color: white; 
                    text-decoration: none; 
                    font-weight: 600;
                ">
                    /yasminauliaaa
                </a>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Additional info
    st.markdown("### 💡 What I Can Help With:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 📊 **Data Analysis & Visualization**
        - 🤖 **Machine Learning Projects**
        - 📈 **Business Intelligence Solutions**
        """)
    
    with col2:
        st.markdown("""
        - 🔍 **Exploratory Data Analysis**
        - 🎯 **Predictive Modeling**
        - 🚀 **End-to-End ML Deployment**
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #888;">
            <p style="font-size: 1.1rem;">
                ✨ Thank you for visiting this application! ✨
            </p>
            <p style="margin-top: 1rem;">
                Made with ❤️ using Streamlit
            </p>
        </div>
    """, unsafe_allow_html=True)